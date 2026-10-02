from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Union,
)

import zmq
from torch.distributed import barrier

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
)
from sglang.srt.managers.mm_utils import (
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.runtime_context import get_disagg, get_parallel, is_ep_scale_joiner
from sglang.srt.utils import (
    broadcast_pyobj,
    point_to_point_pyobj,
)
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.rust_server import RustServer
    from sglang.srt.server_args import ServerArgs
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerRequestReceiver:
    """ Scheduler 的组件，负责接收从 TokenizerManager 发送来的消息
    """

    # NOTE
    #【主链路】接收 TokenizerManager 的消息
    # - 对端：TokenizerManager；开 DP（Data Parallelism）时是 DataParallelController
    # - socket 类型：PULL（只收）
    # - 消息内容：推理请求 TokenizedGenerateReqInput，以及 abort、flush cache 等控制消息
    # - 回复：推理结果发给 DetokenizerManager；控制消息的回复走 send_to_tokenizer
    recv_from_tokenizer: Union[zmq.Socket, ScriptedTokenizerRecvProxy, RustServer]
    # 接收 Engine 的管理命令；DEALER（可收可发）
    # - 消息内容：管理类 RPC（Remote Procedure Call），比如 collective_rpc
    # - 回复：RpcReqOutput 从同一个 socket 发回 Engine
    recv_from_rpc: Optional[zmq.Socket]
    recv_skipper: Any
    input_blocker: Any
    mm_receiver: Any
    ps: ParallelState
    tp_group: Any
    tp_cpu_group: Any
    attn_tp_group: Any
    attn_tp_cpu_group: Any
    attn_cp_group: Any
    attn_cp_cpu_group: Any
    world_group: Any
    server_args: ServerArgs
    model_config: ModelConfig
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_batch: Callable[[], Any]
    scripted_scheduler_hook: Optional[ScriptedSchedulerHook] = None

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0: # 没有配置则直接返回false
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    @scheduler_nvtx_method("scheduler.recv_requests")
    def recv_requests(
        self,
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks.

            - TP: Tensor Parallelism，把模型每一层的权重切开分到多张 GPU，横向拆分
            - PP：Pipeline Parallelism 是纵向拆分
            - tp_rank：TP 组里每张卡的编号，从 0 开始。每张卡对应一个 Scheduler 进程
        """

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        if self.recv_skipper is not None:
            if not self.recv_skipper.handle(self.get_last_batch()):
                return []

        # NOTE 接收消息
        # 接收 TokenizerManager 发送的请求参数：token、采样参数等。
        recv_reqs = self._pull_raw_reqs()

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        # NOTE x_rank=0的卡接收到消息后广播给其他GPU卡
        # rank 0 把收到的完整请求列表原样复制给其他 rank，其他rank收到请求但仅计算自己的一份
        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs)

        # pipeline parallelism 逻辑走这里，如果没有配置则所有 scheduler 节点 pp_rank 都是0
        if self.ps.pp_rank == 0:
            # 把请求里被打包成字节的字段还原成 Python 对象
            self.unwrap_pickle_wrapper(recv_reqs)
        # 多模态相关处理，视频、图片等
        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs

    def _pull_raw_reqs(self) -> Optional[List]:
        """拉取 TokenizerManager 发来的原始请求（还没广播）。
            tp_rank 和 pp_rank 是两种独立的 gpu卡编号，分别是横向和纵向的,
            cp(Data Parallelism) 是对输入进行拆分
            NOTE (pp_rank, tp_rank)类似坐标，唯一确定一张卡和一份权重
             - tp_rank：卡在本级的 TP 组里排第几，也就是负责每层的哪一份权重，属于横向层内切
             - pp_rank：卡在第几级流水线，也就是负责哪一段层，属于纵向按层切
             - cp_rank：Data Parallelism，也是 cp_rank=0 接收信息，然后广播给其他节点
            详见 ../../../../../user_guide_zh/核心概念一：SGLang中的并行策略.md
        """
        if self.ps.pp_rank == 0:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                # 如果是跟 TokenizerManager 通信的GPU卡
                # TODO：https://lmsysorg.mintlify.app/docs/advanced_features/prefill_cp
                recv_reqs = []

                # Rust ringbuffer backend: drain the in-process ring fed by the
                # embedded Rust TokenizerManager instead of a zmq socket. Same
                # non-blocking, msgpack-decoded contract as the zmq path below.
                if envs.SGLANG_RUST_SERVER.get():
                    recv_reqs.extend(
                        self.recv_from_tokenizer.drain(self.max_recv_per_poll)
                    )
                    return recv_reqs

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        # NOTE 接收 TokenizerManager 的推理任务消息
                        #   这里接受到的推理任务消息包括所有的输入，
                        #   注意、attn_cp_rank 是gpu卡属性
                        #       cp_rank=0 的卡：去 ZMQ 管道里读消息。
                        #       cp_rank=1 的卡：后面通过广播拿到同样的消息。
                        recv_req = sock_recv(self.recv_from_tokenizer, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_req)

                while True:
                    try:
                        if self.recv_limit_reached(len(recv_reqs)):
                            break
                        # recv_from_rpc 接收 Engine 的管理命令
                        recv_rpc = sock_recv(self.recv_from_rpc, zmq.NOBLOCK)
                    except zmq.ZMQError:
                        break
                    recv_reqs.append(recv_rpc)
            else:
                recv_reqs = None
        else:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                # 只有开启 Pipeline Parallelism 时才会执行
                # 在if self.ps.pp_rank == 0 的 else 链路中
                dp_offset = (
                    self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
                )
                recv_reqs = point_to_point_pyobj(
                    [],
                    self.ps.pp_rank * self.ps.tp_size + dp_offset,
                    self.world_group.cpu_group,
                    (self.ps.pp_rank - 1) * self.ps.tp_size + dp_offset,
                    self.ps.pp_rank * self.ps.tp_size + dp_offset,
                )
            else:
                recv_reqs = None
        return recv_reqs

    def _broadcast_reqs_across_ranks(self, recv_reqs: Optional[List]) -> List:
        if get_parallel().enable_dp_attention:
            if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
            else:
                work_reqs = None
                control_reqs = None

            if self.ps.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )

            if self.ps.attn_cp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_cp_group.rank,
                    self.attn_cp_cpu_group,
                    src=self.attn_cp_group.ranks[0],
                )

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            _local_ctrl = (
                get_parallel().enable_dp_attention_local_control_broadcast
                or is_ep_scale_joiner()
            )
            if _local_ctrl:
                if self.ps.attn_tp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_tp_group.rank,
                        self.attn_tp_cpu_group,
                        src=self.attn_tp_group.ranks[0],
                    )
                if self.ps.attn_cp_size != 1:
                    control_reqs = broadcast_pyobj(
                        control_reqs,
                        self.attn_cp_group.rank,
                        self.attn_cp_cpu_group,
                        src=self.attn_cp_group.ranks[0],
                    )
            elif self.ps.tp_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        elif self.ps.tp_size != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return recv_reqs

    def unwrap_pickle_wrapper(self, recv_reqs: Optional[List]) -> None:
        if not recv_reqs:
            return

        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                req.unwrap_pickle_fields()
            elif isinstance(
                req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
            ):
                for sub_req in req:
                    sub_req.unwrap_pickle_fields()

    def _apply_mm_receiver(self, recv_reqs: List) -> List:
        # Process MM requests under EPD-disaggregation mode
        if (
            self.ps.pp_rank == 0
            and get_disagg().language_only
            and get_disagg().encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
            for req, error_msg, error_code in abort_reqs:
                if error_code is None:
                    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
                elif isinstance(error_code, HTTPStatus):
                    status_code = error_code
                else:
                    status_code = HTTPStatus(int(error_code))
                prepare_abort(req, error_msg, status_code=status_code)
                self.stream_output([req], req.return_logprob)
        return recv_reqs

    def _finalize_shm_features(self, recv_reqs: Optional[List]) -> None:
        # Unwrap shared memory features AFTER all broadcasts complete,
        # so that ShmPointerMMData metadata (not full tensor data) is what
        # gets serialized during broadcast_pyobj.
        # “把多模态请求里"指向共享内存的指针"换成真正的数据”
        if recv_reqs:
            if self.model_config.is_multimodal and has_shm_features(recv_reqs):
                # The broadcast source returns with its original objects while
                # peer ranks may still be unpickling ShmPointerMMData
                # (-> shm_open).  Synchronize the same CPU groups that carried
                # SHM-backed work requests before materialize() unlinks them.
                if get_parallel().enable_dp_attention:
                    if self.ps.attn_tp_size > 1:
                        barrier(group=self.attn_tp_cpu_group)
                    if self.ps.attn_cp_size > 1:
                        barrier(group=self.attn_cp_cpu_group)
                elif self.ps.tp_size > 1:
                    barrier(group=self.tp_cpu_group)
            for req in recv_reqs:
                unwrap_shm_features(req)

    def _split_work_and_control_reqs(self, recv_reqs: List):
        work_reqs = [
            req
            for req in recv_reqs
            if isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        control_reqs = [
            req
            for req in recv_reqs
            if not isinstance(
                req,
                (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    BatchTokenizedGenerateReqInput,
                    BatchTokenizedEmbeddingReqInput,
                ),
            )
        ]
        return work_reqs, control_reqs
