# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DetokenizerManager is a process that detokenizes the token ids."""

import dataclasses
import logging
import os
import signal
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import psutil
import pybase64
import setproctitle
import torch
import zmq

from sglang.srt.beam_search.output import (
    decode_beam_search_output,
    is_beam_search_batch,
)
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    ConfigureLoggingReq,
    FreezeGCReq,
    sock_recv,
    sock_send,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiHttpWorkerDetokenizerMixin
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.runtime_context import (
    get_device,
    get_model,
    get_observability,
    get_serving,
    publish,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, freeze_gc, kill_itself_when_parent_died
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.network import get_zmq_socket
from sglang.srt.utils.patch_tokenizer import decode_without_hf_kwargs
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import (
    TypeBasedDispatcher,
    find_printable_text,
    get_exception_traceback,
)

logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


@dataclasses.dataclass
class DecodeStatus:
    """
        Store the status of incremental(increases in value or worth, often by a regular amount) decoding.
        GPU 输出 token ID → 追加到 decode_ids → 解码成新文本放进 decoded_text_chunks → 请求结束时合并进 decoded_text
    """

    # 用输入“你数三个数” 作示例

    # tokenID，包括两部分：
    #   1. 你数三个数 对应的输入 tokenID
    #   2. 输出的tokenID
    decode_ids: List[int] #【重点】
    surr_offset: int  # 上下文窗口起点
    read_offset: int  # 新 token 起点

    # 没有默认值，dataclass 要求它排在有默认值的 decoded_text_chunks 前面
    decoded_text: str # 和输出tokenID对应的文本，比如输出可能依次是： 1，然后是 12，然后是 123 #【重点】
    decoded_text_chunks: List[str] = dataclasses.field(default_factory=list) #【重点】
    decoded_text_len: int = dataclasses.field(init=False)

    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0  # 已发给 TokenizerManager 的字符数，防止重复发送

    def __post_init__(self):
        self.decoded_text_len = len(self.decoded_text)

    def append_decoded_text(self, text: str):
        if text:
            self.decoded_text_chunks.append(text)
            self.decoded_text_len += len(text)

    def get_decoded_text(self) -> str:
        if self.decoded_text_chunks:
            self.decoded_text += "".join(self.decoded_text_chunks)
            self.decoded_text_chunks.clear()
        return self.decoded_text


class DetokenizerManager(MultiHttpWorkerDetokenizerMixin):
    """DetokenizerManager is a process that detokenizes the token ids.
       独立进程，它负责
        1. 从 Scheduler 接收生成的 token ID: recv_from_scheduler
        2. 按请求增量解码成新文本（嵌入结果直接透传）
        3. 把结果发回 TokenizerManager
       详见 ../../../../user_guide_zh/核心组件三：DetokenizerManager.md
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # 【重要】交互参考
        # Init inter-process communication：两个管道：
        #   Scheduler - recv_from_scheduler -> DetoenizerManager：接收gpu推理结果的tokenID
        #   DetoenizerManager - ID，send_to_tokenizer -> TokenizerManager：将tokenID解码后的结果文本给到 TokenizerManager 
        self.init_ipc_channels(port_args, server_args)

        # Init tokenizer
        # 加载同TokenizerManager一份分词器
        self.init_tokenizer(server_args)

        # Init running status “进入主循环前，把运行时要用的状态和开关准备好”
        self.init_running_status(server_args)

        # Init dispatcher
        self.init_request_dispatcher()

    def init_ipc_channels(self, port_args: PortArgs, server_args: ServerArgs):
        context = zmq.Context(2)
        # Scheduler 进程                                             Detokenizer 进程
        # PUSH socket    ---- token ID（detokenizer_ipc_name）---->  PULL socket（bind）
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        # In multi-tokenizer mode, results are pushed back to each TokenizerWorker
        # directly via SocketMapping inside multi_http_worker_event_loop, so the
        # single send_to_tokenizer socket is unused.
        if server_args.tokenizer_worker_num == 1:
            # Detokenizer 进程                                       TokenizerManager 进程
            # PUSH socket（connect）--- 文本（tokenizer_ipc_name）---> PULL socket（bind）
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )

    def init_tokenizer(self, server_args: ServerArgs):
        """ 加载 tokenizer 之后只用 decode
        """
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
            self.vocab_size = None
        else:
            self.tokenizer = get_tokenizer(
                get_serving().tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=get_model().trust_remote_code,
                revision=server_args.revision,
                tokenizer_backend=server_args.tokenizer_backend,
            )
            try:
                self.vocab_size = len(self.tokenizer)
            except TypeError:
                self.vocab_size = getattr(self.tokenizer, "vocab_size", None)

    def init_running_status(self, server_args: ServerArgs):
        # 【重要】初始化一个有序、有容量限制的字典，保存 请求 id 到结果的映射
        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)

        self.disable_tokenizer_batch_decode = server_args.disable_tokenizer_batch_decode
        self.is_tool_call_parser_gpt_oss = get_serving().tool_call_parser == "gpt-oss"

        self.soft_watchdog = Watchdog.create(
            debug_name="DetokenizerManager",
            watchdog_timeout=get_device().soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DETOKENIZER.get(),
        )

        if get_observability().enable_metrics:
            start_cpu_monitor_thread("detokenizer")

    def init_request_dispatcher(self):
        # 对于 Scheduler 的消息，不同类型消息使用不同的处理方法处理
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOutput, self.handle_batch_embedding_out),
                (BatchTokenIDOutput, self.handle_batch_token_id_out),  # 【重点】生成任务走这里
                (FreezeGCReq, self.handle_freeze_gc_req),
                (ConfigureLoggingReq, self.handle_configure_logging_req),
            ]
        )

    def event_loop(self):
        """The event loop that handles requests
        【重点】从 scheduler 获取 tokenID 任务以及将结果发送给 TokenizerManager 都在这个类方法中
            单 tokenizer worker 模式的主循环；
            多 worker 模式走 multi_http_worker_event_loop
        """
        while True:
            # step 1
            # 【重点】阻塞获取 Scheduler 的消息
            with self.soft_watchdog.disable():
                recv_obj = sock_recv(self.recv_from_scheduler) 
           
            # step 2
            # _request_dispatcher 保存了不同的数据类型对应的处理方式
            # BatchTokenIDOutput 对应 handle_batch_token_id_out
            output = self._request_dispatcher(recv_obj)

            # step 3
            # 结果发回 TokenizerManager，由 TokenizerManager#handle_loop 接收
            if output is not None:
                sock_send(self.send_to_tokenizer, output)
            self.soft_watchdog.feed()

    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            if pos == -1:
                return output
            end = pos + len(matched)
            return output[:end] if no_stop_trim else output[:pos]

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            if no_stop_trim:
                return output
            # 200012 <|call|> is the tool call token and one of eos tokens for gpt-oss model
            if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
                return output
            assert len(output) > 0
            # NOTE: We can always assume the last token is the matched stop token
            return output[:-1]
        return output

    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOutput):
        # If it is embedding model, no detokenization is needed.
        # 嵌入任务没有文本要解码，原样转发给 TokenizerManager
        return recv_obj

    @staticmethod
    def _clamp_decode_ids(ids: List[int], vocab_size: Optional[int]) -> List[int]:
        """Map out-of-range token ids to 0 so the tokenizer can decode them.
        把不在词表范围内的 token ID（负数，或者大于等于词表大小的）换成 0，免得分词器解码时报错

        Multimodal placeholder ids (e.g. Inkling's negative -101/-102, or radix-cache
        pad-value hashes) are not real vocab tokens; tiktoken-style backends raise
        OverflowError on negative / out-of-range ids. These only appear in the
        surrogate-context prefix (before read_offset) and carry no text, and the clamp
        is applied identically to surr_ids and read_ids, so the incremental
        (read-minus-surr) output text is unchanged.
        """
        hi = vocab_size if vocab_size else None
        return [t if (0 <= t and (hi is None or t < hi)) else 0 for t in ids]

    def _grouped_batch_decode(
        self,
        ids_list: List[List[int]],
        skip_list: List[bool],
        space_list: List[bool],
    ) -> List[str]:
        """Batch decode with grouping by (skip_special_tokens, spaces_between_special_tokens).
        """
        n = len(ids_list)
        if n == 0:
            return []
        logger.info(
            f"[detok-debug] 输入 n={n} ids_list={ids_list} skip={skip_list} space={space_list}"
        )

        # Empty token spans decode to "" but tokenizer.batch_decode (and the
        # slow per-row decode_without_hf_kwargs path) still pays per-row
        # overhead; under high-concurrency streaming this adds up. Filter
        # empties out, decode the rest, then scatter back.
        keep_idx: Optional[List[int]] = None
        if not all(ids_list):
            keep_idx = [i for i, ids in enumerate(ids_list) if ids]
            if not keep_idx:
                logger.info(f"[detok-debug] 全部为空，直接返回 {n} 个空串")
                return [""] * n
            ids_list = [ids_list[i] for i in keep_idx]
            skip_list = [skip_list[i] for i in keep_idx]
            space_list = [space_list[i] for i in keep_idx]
        logger.info(f"[detok-debug] keep_idx={keep_idx}（None 表示没有空列表）")

        if not getattr(self.tokenizer, "is_fast", False):
            logger.info("[detok-debug] 分支：slow tokenizer，逐条 decode_without_hf_kwargs")
            decoded = [
                decode_without_hf_kwargs(self.tokenizer, ids, skip)
                for ids, skip in zip(ids_list, skip_list)
            ]
        else:
            # fast path: all rows share the same (skip, space) flags. 【重点】走这里
            first_skip, first_space = skip_list[0], space_list[0]
            if all(
                s == first_skip and sp == first_space
                for s, sp in zip(skip_list, space_list)
            ):
                # 【重要】【重要】【重要】结果 tokenID -> 输出文本
                logger.info(
                    f"[detok-debug] 分支：fast，选项相同，一次 batch_decode（skip={first_skip}, space={first_space}）"
                )
                decoded = self.tokenizer.batch_decode(
                    ids_list,
                    skip_special_tokens=first_skip,
                    spaces_between_special_tokens=first_space,
                )
            else:
                # Group indices by (skip, space) tuple and decode each group.
                groups: Dict[Tuple[bool, bool], List[int]] = defaultdict(list)
                for idx, (skip, space) in enumerate(zip(skip_list, space_list)):
                    groups[(skip, space)].append(idx)
                logger.info(f"[detok-debug] 分支：fast，选项不同，按组 batch_decode groups={dict(groups)}")

                decoded = [""] * len(ids_list)
                for (skip, space), indices in groups.items():
                    group_decoded = self.tokenizer.batch_decode(
                        [ids_list[idx] for idx in indices],
                        skip_special_tokens=skip,
                        spaces_between_special_tokens=space,
                    )
                    for idx, text in zip(indices, group_decoded):
                        decoded[idx] = text

        if keep_idx is None:
            logger.info(f"[detok-debug] 输出 results={decoded}")
            return decoded
        results = [""] * n
        for i, text in zip(keep_idx, decoded):
            results[i] = text
        logger.info(f"[detok-debug] 输出 decoded={decoded} → 放回原位 results={results}")
        return results

    def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
        """增量解码，surr_ids / read_ids 的用法详见 detokenizer_manager.py._decode_batch_token_id_output.md"""
        bs = len(recv_obj.rids)
        vocab_size = self.vocab_size

        # Initialize decode status
        # 【Step 2.1】维护 decode_status[rid]：取或建 DecodeStatus，追加本次新 token；Scheduler 每次只发新增的部分
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            # 每个请求第一个结果token返回回走到这个分支
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i], # recv_obj.decoded_texts[i] 在这里还是空字符串
                    decode_ids=self._clamp_decode_ids( # 对 decode_ids 做无害处理后赋值给 DecodeStatus
                        recv_obj.decode_ids[i], vocab_size
                    ),
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s # 【重要】创建 rid 到结果的映射
            else:
                s = self.decode_status[rid]
                s.decode_ids.extend( # 已有该 rid 的 DecodeStatus：把本次新 token ID 追加到 decode_ids 末尾，后面基于它做增量解码
                    self._clamp_decode_ids(recv_obj.decode_ids[i], vocab_size)
                )

            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

        # Decode token ids to strings
        # 【Step 2.2】整批解码 surr_ids 和 read_ids
        if not self.disable_tokenizer_batch_decode:
            # 【重要】默认走 batch
            surr_texts = self._grouped_batch_decode(
                surr_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
            read_texts = self._grouped_batch_decode(
                read_ids,
                recv_obj.skip_special_tokens,
                recv_obj.spaces_between_special_tokens,
            )
        else:
            # Do not use batch decode to prevent some detokenization edge cases (e.g., gpt-oss).
            surr_texts = [
                self.tokenizer.decode(
                    surr, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for surr, skip, space in zip(
                    surr_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]
            read_texts = [
                self.tokenizer.decode(
                    read, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for read, skip, space in zip(
                    read_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]

        # Incremental decoding
        # 【Step 2.3】取差值得到新文本；未结束时推进偏移，结束时删除状态并补发剩余文本
        output_strs = []
        for i in range(bs):
            rid = recv_obj.rids[i]
            try:
                s = self.decode_status[rid]
            except KeyError:
                raise RuntimeError(
                    f"Decode status not found for request {rid}. "
                    "It may be due to the request being evicted from the decode status due to memory pressure. "
                    "Please increase the maximum number of requests by setting "
                    "the SGLANG_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                    f"The current value is {DETOKENIZER_MAX_STATES}. "
                    "For more details, see: https://github.com/sgl-project/sglang/issues/2812"
                )
            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming. Invariant: sent_offset >= decoded_text_len. The
                # gap (`pending`) is "printable but uncommitted" text emitted
                # in a prior "�" recovery step; we skip it from this step's
                # emission so we don't double-send.
                pending = s.sent_offset - s.decoded_text_len
                if new_text and not new_text.endswith("�"):
                    # Clean text: commit to decoded_text and advance offsets.
                    s.append_decoded_text(new_text)
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    s.sent_offset = s.decoded_text_len
                    output_strs.append(new_text[pending:] if pending else new_text)
                else:
                    # Incomplete UTF-8: emit the printable prefix only; do not
                    # commit (token offsets stay so the next iteration retries
                    # with more tokens).
                    printable = find_printable_text(new_text)
                    s.sent_offset = s.decoded_text_len + len(printable)
                    output_strs.append(printable[pending:] if pending else printable)
                continue

            if rid in self.decode_status:
                del self.decode_status[rid]

            # Finished: materialize once, trim the matched stop, emit the tail.
            output_str = self.trim_matched_stop(
                s.get_decoded_text() + new_text,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            output_strs.append(incremental_output)

        return output_strs

    @staticmethod
    def _b64_encode_per_request(
        data_list: Optional[List[Optional[torch.Tensor]]],
    ) -> Optional[List[Optional[str]]]:
        """Encode a per-request list of tensors as base64 strings, off the
        tokenizer hot path. Returns None when the input is None; per-item None
        stays None.
        """
        if data_list is None:
            return None
        return [
            (
                pybase64.b64encode(item.numpy().tobytes()).decode("utf-8")
                if item is not None
                else None
            )
            for item in data_list
        ]

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # Beam decoding is additive: a batch may mix beam leaders with normal
        # requests, so every item still goes through the standard decode.
        if is_beam_search_batch(recv_obj):
            decode_beam_search_output(
                recv_obj,
                tokenizer=self.tokenizer,
                disable_batch_decode=self.disable_tokenizer_batch_decode,
                trim_matched_stop=self.trim_matched_stop,
            )
        # If handling idle batch, set output_strs to [].
        # 【重点】增量解码，见 _decode_batch_token_id_output
        output_strs = (
            self._decode_batch_token_id_output(recv_obj)
            # 【重要】
            #   rid 是request id、请求唯一标识
            #   list 是因为兼容批处理
            if len(recv_obj.rids) > 0
            else []
        )
        routed_experts = self._b64_encode_per_request(recv_obj.routed_experts)
        indexer_topk = self._b64_encode_per_request(recv_obj.indexer_topk)
        # 只有 output_strs 是这里算出来的，其余字段（token 计数、logprob 等）基本原样透传
        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            reasoning_tokens=recv_obj.reasoning_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            cached_tokens_details=recv_obj.cached_tokens_details,
            image_tokens=recv_obj.image_tokens,
            audio_tokens=recv_obj.audio_tokens,
            video_tokens=recv_obj.video_tokens,
            spec_verify_ct=recv_obj.spec_verify_ct,
            spec_num_correct_drafts=recv_obj.spec_num_correct_drafts,
            spec_num_block_accept_tokens=recv_obj.spec_num_block_accept_tokens,
            spec_num_cap_tokens=recv_obj.spec_num_cap_tokens,
            spec_correct_drafts_histogram=recv_obj.spec_correct_drafts_histogram,
            spec_cap_lens_histogram=recv_obj.spec_cap_lens_histogram,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            input_top_logprobs_val_flat=recv_obj.input_top_logprobs_val_flat,
            input_top_logprobs_idx_flat=recv_obj.input_top_logprobs_idx_flat,
            input_top_logprobs_flat_null_prefix=recv_obj.input_top_logprobs_flat_null_prefix,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_token_entropy_val=recv_obj.output_token_entropy_val,
            output_token_sampling_mask=recv_obj.output_token_sampling_mask,
            output_token_sampling_logprobs=recv_obj.output_token_sampling_logprobs,
            output_hidden_states=recv_obj.output_hidden_states,
            routed_experts=routed_experts,
            indexer_topk=indexer_topk,
            customized_info=recv_obj.customized_info,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=recv_obj.retraction_counts,
            weight_versions=recv_obj.weight_versions,
            token_steps=recv_obj.token_steps,
            beam_search_output=recv_obj.beam_search_output,
            dp_ranks=recv_obj.dp_ranks,
            time_stats=recv_obj.time_stats,
        )

    def handle_freeze_gc_req(self, recv_req: FreezeGCReq):
        freeze_gc("Detokenizer Manager")
        return None

    def handle_configure_logging_req(self, recv_req: ConfigureLoggingReq):
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())


def is_health_check_request(rid: Optional[str]) -> bool:
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


class LimitedCapacityDict(OrderedDict):
    """ 有容量限制、有序的 字典
        满了之后再插入新项，会先淘汰最早放进去的那一项
    """
    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    detokenizer_manager_class=DetokenizerManager,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    publish(server_args, role="detokenizer")
    parent_process = psutil.Process().parent()

    manager = None
    try:
        # 【启动】Engine._launch_detokenizer_subprocesses 用 mp.Process 启动本函数，在新进程里构造 DetokenizerManager
        manager = detokenizer_manager_class(server_args, port_args)
        # 进入主循环，不再返回
        if server_args.tokenizer_worker_num == 1:
            manager.event_loop()
        else:
            manager.multi_http_worker_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        if manager is not None:
            manager.maybe_clear_socket_mapping()
        parent_process.send_signal(signal.SIGQUIT)
