from __future__ import annotations

import logging
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

import torch
import zmq
from torch.distributed import ReduceOp, all_reduce, barrier

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.distributed.communication_op import attn_cp_tp_broadcast_pyobj
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    MMInputsProcessError,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
)
from sglang.srt.managers.mm_utils import (
    discard_shm_features,
    has_shm_features,
    unwrap_shm_features,
)
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_RECV_REQUESTS,
    SchedulerStageMetricsRecorder,
    scheduler_stage_method,
)
from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel
from sglang.srt.utils import (
    broadcast_pyobj,
    point_to_point_pyobj,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.rust_server.server import RustServer
    from sglang.srt.server_args import ServerArgs
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
    from sglang.test.scripted_runtime.tokenizer_recv_proxy import (
        ScriptedTokenizerRecvProxy,
    )

logger = logging.getLogger(__name__)


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
    server_args: ServerArgs
    model_config: ModelConfig
    max_recv_per_poll: int
    stream_output: Callable[..., None]
    get_last_batch: Callable[[], Any]
    scripted_scheduler_hook: Optional[ScriptedSchedulerHook] = None
    scheduler_stage_metrics: Optional[SchedulerStageMetricsRecorder] = None

    def recv_limit_reached(self, num_recv_reqs: int) -> bool:
        if self.max_recv_per_poll < 0: # 没有配置则直接返回false
            return False
        return num_recv_reqs >= self.max_recv_per_poll

    @scheduler_stage_method(SCHEDULER_STAGE_RECV_REQUESTS)
    def recv_requests(
        self, local_reqs: Optional[List[AbortReq]] = None
    ) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
        """Receive results at tp_rank = 0 and broadcast it to all other TP ranks.
        NOTE 由接收端 rank 拉取 TokenizerManager 的请求，再广播给其他并行 rank。

        local_reqs are aborts the caller decided on this rank; they ride the
        same broadcast as the pulled requests.
        """

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.step()

        if self.recv_skipper is not None:
            if not self.recv_skipper.handle(self.get_last_batch()):
                return []

        # NOTE 1）接收消息
        # 接收 TokenizerManager 发送的请求参数：token、采样参数等
        #   如果没有请求， recv_req 则为空
        recv_reqs = self._pull_raw_reqs()
        if recv_reqs:
            logger.info("[inf] [Scheduler] recv_requests, recv_reqs:\n%s", recv_reqs)

        if self.input_blocker is not None:
            recv_reqs = self.input_blocker.handle(recv_reqs)

        # NOTE 广播完整请求，各 GPU 收到请求后按并行分工计算；local_reqs 同步本地中止请求。
        recv_reqs = self._broadcast_reqs_across_ranks(recv_reqs, local_reqs)

        if get_parallel().pp_rank == 0:
            # 把请求里被打包成字节的字段还原成 Python 对象。
            self.unwrap_pickle_wrapper(recv_reqs)
        # 多模态相关处理，视频、图片等
        recv_reqs = self._apply_mm_receiver(recv_reqs)

        self._finalize_shm_features(recv_reqs)

        return recv_reqs

    def _pull_raw_reqs(self) -> Optional[List]:
        """拉取 TokenizerManager 或上一流水线阶段发来的原始请求。"""
        if get_parallel().pp_rank == 0:
            if get_parallel().attn_tp_rank == 0 and get_parallel().attn_cp_rank == 0:
                # 第一流水线阶段由注意力 TP/CP 的 rank 0 接收 TokenizerManager 消息。
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
                        recv_req = sock_recv(self.recv_from_tokenizer, zmq.NOBLOCK)
                        logger.info("[inf] [Scheduler] _pull_raw_reqs, recv_req:\n%s", recv_req)
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
            if get_parallel().attn_tp_rank == 0 and get_parallel().attn_cp_rank == 0:
                # 非首个流水线并行(Pipeline Parallelism)阶段从上一阶段接收。
                dp_offset = (
                    get_parallel().attn_dp_rank
                    * get_parallel().attn_cp_size
                    * get_parallel().attn_tp_size
                )
                recv_reqs = point_to_point_pyobj(
                    [],
                    get_parallel().pp_rank * get_parallel().tp_size + dp_offset,
                    get_parallel().world_group.cpu_group,
                    (get_parallel().pp_rank - 1) * get_parallel().tp_size + dp_offset,
                    get_parallel().pp_rank * get_parallel().tp_size + dp_offset,
                )
            else:
                recv_reqs = None
        return recv_reqs

    def _broadcast_reqs_across_ranks(
        self, recv_reqs: Optional[List], local_reqs: Optional[List] = None
    ) -> List:
        """local_reqs ride the work channel, which is scoped to the ranks
        sharing one waiting queue; the control channel fans out from global
        rank 0 and would overwrite every DP group's aborts but the first.
        """
        local_reqs = local_reqs or []
        if get_parallel().attn_dp_enabled:
            if get_parallel().attn_tp_rank == 0 and get_parallel().attn_cp_rank == 0:
                work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
                work_reqs.extend(local_reqs)
            else:
                work_reqs = None
                control_reqs = None

            work_reqs = attn_cp_tp_broadcast_pyobj(work_reqs)

            # When dp_attention_local_control_broadcast is enabled, each DP
            # group leader already receives control messages from the DP
            # controller, so we broadcast within attn_tp_group + attn_cp_group
            # instead of the full tp_group.  This avoids an expensive
            # all-ranks gloo sync.
            _local_ctrl = (
                get_parallel().enable_dp_attention_local_control_broadcast
                or get_exec().moe.is_ep_scale_joiner
            )
            if _local_ctrl:
                control_reqs = attn_cp_tp_broadcast_pyobj(control_reqs)
            elif get_parallel().tp_size != 1:
                tp_group = get_parallel().tp_group
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    tp_group.rank,
                    tp_group.cpu_group,
                    src=tp_group.ranks[0],
                )
            recv_reqs = work_reqs + control_reqs
        else:
            if recv_reqs is not None:
                recv_reqs = [*recv_reqs, *local_reqs]
            if get_parallel().tp_size != 1:
                tp_group = get_parallel().tp_group
                recv_reqs = broadcast_pyobj(
                    recv_reqs,
                    tp_group.rank,
                    tp_group.cpu_group,
                    src=tp_group.ranks[0],
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
            get_parallel().pp_rank == 0
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
        """Materialize SHM features or mark the request failed on every rank."""
        # 将多模态请求里的共享 CPU 内存(CPU shared memory)引用还原为可用特征。
        if not recv_reqs or not self.model_config.is_multimodal:
            return

        tokenized_reqs = []
        for req in recv_reqs:
            if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
                tokenized_reqs.append(req)
            elif isinstance(
                req,
                (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput),
            ):
                tokenized_reqs.extend(req.batch)
        if not tokenized_reqs or not has_shm_features(tokenized_reqs):
            return

        # 1. wait until every rank has opened the shared feature segments
        parallel = get_parallel()
        if parallel.attn_dp_enabled:
            if parallel.attn_tp_size > 1:
                barrier(group=parallel.attn_tp_group.cpu_group)
            if parallel.attn_cp_size > 1:
                barrier(group=parallel.attn_cp_group.cpu_group)
        elif parallel.tp_size > 1:
            barrier(group=parallel.tp_group.cpu_group)

        # 2. materialize independently so one bad VLM request does not stop the loop
        failed = torch.zeros(len(tokenized_reqs), dtype=torch.int32)
        for index, req in enumerate(tokenized_reqs):
            if not has_shm_features([req]):
                continue
            try:
                unwrap_shm_features(req)
            except Exception:
                logger.exception(
                    "Failed to materialize shared-memory multimodal features for rid=%s",
                    req.rid,
                )
                discard_shm_features(req)
                failed[index] = 1

        # 3. all ranks reject the same requests before entering model collectives
        if parallel.attn_dp_enabled:
            if parallel.attn_tp_size > 1:
                all_reduce(
                    failed, op=ReduceOp.MAX, group=parallel.attn_tp_group.cpu_group
                )
            if parallel.attn_cp_size > 1:
                all_reduce(
                    failed, op=ReduceOp.MAX, group=parallel.attn_cp_group.cpu_group
                )
        elif parallel.tp_size > 1:
            all_reduce(failed, op=ReduceOp.MAX, group=parallel.tp_group.cpu_group)

        error = MMInputsProcessError(
            "Failed to materialize shared-memory multimodal features on a scheduler rank."
        )
        for index, req in enumerate(tokenized_reqs):
            if failed[index].item():
                discard_shm_features(req)
                req.mm_inputs = error

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
