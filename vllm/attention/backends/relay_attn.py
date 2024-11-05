"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         BlockDiagonalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import (PAD_SLOT_ID, CommonAttentionState,
                                           compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.backends.functional.flashattn_ops import flash_attn_with_kvcache
from vllm.attention.backends.functional.relayattn_ops import relay_fusion
from vllm.attention.backends.functional.xformers_ops import memory_efficient_attention_forward

from vllm.forward_context import get_forward_context
from vllm.utils import async_tensor_h2d, make_tensor_with_pad

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)

_PARTITION_SIZE = 512

class RelayAttentionBackend(AttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "RELAY_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["RelayAttentionImpl"]:
        return RelayAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return RelayAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["RelayAttentionMetadataBuilder"]:
        return RelayAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class RelayAttentionMetadata(AttentionMetadata):
    """Metadata for RelayAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum query length in the batch.
    max_query_len: Optional[int]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    is_prompt: bool
    prefix_length: int
    prefix_length_buffer: Optional[torch.Tensor]

    _cached_prefill_metadata: Optional["RelayAttentionMetadata"] = None
    _cached_decode_metadata: Optional["RelayAttentionMetadata"] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None

    @property
    def prefill_metadata(self) -> Optional["RelayAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.block_tables is not None

        self._cached_prefill_metadata = RelayAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            max_query_len=self.max_query_len,
            block_tables=self.block_tables[:self.num_prefills],
            prefix_length = 0,
            prefix_length_buffer = None,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["RelayAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = RelayAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            max_query_len=self.max_query_len,
            block_tables=self.block_tables[self.num_prefills:],
            prefix_length = 0,
            prefix_length_buffer = None,
        )
        return self._cached_decode_metadata

class RelayAttentionMetadataBuilder(
        AttentionMetadataBuilder[RelayAttentionMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)

            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

    def _get_graph_runner_block_tables(
            self, num_seqs: int,
            block_tables: List[List[int]]) -> torch.Tensor:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    graph_block_tables[
                        i, :max_blocks] = block_table[:max_blocks]

        return torch.from_numpy(graph_block_tables).to(
            device=self.runner.device, non_blocking=True)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                prefix_cache_hit)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device,
                                             self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        return RelayAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )


class RelayAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "RelayAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = RelayAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RelayAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
        prefix_kv_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # fill the prefix kv cache here
        # NOTE: this happens only when we fill prefix kv cache
        # by passing input_metadata.prefix_length=-1 as the signal
        if attn_metadata.prefix_length < 0:
            assert (kv_cache is None) and (prefix_kv_cache is not None)
            assert attn_metadata.is_prompt and (len(attn_metadata.seq_lens) == 1)
            seq_len = attn_metadata.seq_lens[0]
            prefix_kv_cache[0, :, :seq_len, :, :] = key.unflatten(-1, (self.num_kv_heads, self.head_size))
            prefix_kv_cache[1, :, :seq_len, :, :] = value.unflatten(-1, (self.num_kv_heads, self.head_size))

        if attn_metadata.prefix_length > 0:
            # FIXME (ray): window attention
            # NOTE: flash attention natively supports MQA/GQA
            output_pre, lse_pre = flash_attn_with_kvcache(
                query.view(1, -1, self.num_heads, self.head_size), # (1, bsz*len, num_heads, head_size)
                k_cache=prefix_kv_cache[0], # (1, prefix_length, num_kv_heads, head_size)
                v_cache=prefix_kv_cache[1], # (1, prefix_length, num_kv_heads, head_size)
                cache_seqlens=attn_metadata.prefix_length_buffer, # (1, )
                softmax_scale=self.scale)
            output_pre:torch.Tensor = output_pre.view(-1, self.num_heads, self.head_size)
            lse_pre:torch.Tensor = lse_pre.squeeze(0)
            trans_lse_pre = True

        num_tokens, hidden_size = query.shape

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size) # (bsz*seqlen, nheads, head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None and (attn_metadata.slot_mapping is not None):
            torch.ops._C_cache_ops.reshape_and_cache(
                key,
                value,
                kv_cache[0],
                kv_cache[1],
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        if attn_metadata.is_prompt:
            # Prompt run.
            if self.num_kv_heads != self.num_heads:
                # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                # project the key and value tensors to the desired number of
                # heads.
                # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                query = query.view(query.shape[0], self.num_kv_heads,
                                   self.num_queries_per_kv, query.shape[-1])
                # query: (bsz*seqlen, self.num_kv_heads, self.num_queries_per_kv, head_size)
                # this is for MQA/GQA, see xops.memory_efficient_attention_forward
                # TODO(ray): use native flash attention
                key = key[:, :, None, :].expand(key.shape[0],
                                                self.num_kv_heads,
                                                self.num_queries_per_kv,
                                                key.shape[-1])
                value = value[:, :, None, :].expand(value.shape[0],
                                                    self.num_kv_heads,
                                                    self.num_queries_per_kv,
                                                    value.shape[-1])

            # TODO(aserson): Add alibi_slopes support
            assert self.alibi_slopes is None

            # Set attention bias if not provided. This typically happens at the
            # very attention layer of every iteration.
            # FIXME(woosuk): This is a hack.
            if attn_metadata.attn_bias is None:
                attn_bias = BlockDiagonalCausalMask.from_seqlens(
                    attn_metadata.seq_lens)
                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                attn_metadata.attn_bias = attn_bias

            # [1, bsz*seqlen, head_groups, num_heads_per_group, K]
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

            out, lse = memory_efficient_attention_forward(query,
                                                          key,
                                                          value,
                                                          attn_bias=attn_metadata.attn_bias,
                                                          p=0.0,
                                                          scale=self.scale)
            output = out.view(num_tokens, self.num_heads, self.head_size)
            # (bsz, head, num_queries) -> (bsz*num_queries, nheads)
            lse = lse.transpose(1, 2).reshape(num_tokens, self.num_heads).contiguous()
        else:
            # Decoding run.
            if kv_cache is not None:
                # (bsz*seqlen, nheads, head_size)
                # (bsz*seqlen, nheads)
                output, lse = _paged_attention(
                    query,
                    kv_cache[0],
                    kv_cache[1],
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens_tensor,
                    attn_metadata.max_decode_seq_len,
                    self.kv_cache_dtype,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                    k_scale,
                    v_scale,
                )
            else:
                # This happens during the initial memory profiling run for
                # CUDA graphs.
                output = torch.zeros_like(query)
                lse = torch.zeros(num_tokens, self.num_heads)

        if attn_metadata.prefix_length > 0:
            # print(output.stride())
            # print(output_pre.stride())
            # print(lse.size(), lse.stride())
            # print(lse_pre.size(), lse_pre.stride())
            # print('------')
            output = relay_fusion(output_pre, lse_pre, output, lse,
                                  backend='triton', trans_lse_sys=trans_lse_pre)
            # output = output

        # print(output.stride())
        # Reshape the output tensor.
        # return output.reshape(num_tokens, hidden_size)
        return output.view(num_tokens, hidden_size)

def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[AttentionBias]:
    attn_biases: List[AttentionBias] = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (seq_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            seq_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :seq_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        if num_heads != num_kv_heads:
            bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases


def _paged_attention(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
        # use blocksparse paged attention
        block_size = value_cache.size(-1)
        assert (blocksparse_block_size > 0 and
                blocksparse_block_size % block_size == 0), \
            (f"{blocksparse_block_size=} needs to be a multiple of"
             f"{block_size=} used in block_tables.")

    output = torch.empty_like(query)
    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                              _PARTITION_SIZE)

    lse = torch.zeros(
        size=(num_seqs, num_heads),
        dtype=torch.float32,
        device=output.device)

    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    # TODO(woosuk): Tune this heuristic.
    # For context len > 8192, use V2 kernel to avoid shared memory shortage.
    use_v1 = (max_seq_len <= 8192
              and (max_num_partitions == 1 or num_seqs * num_heads > 512))

    if use_v1:
        # Run PagedAttention V1.
        ops.paged_attention_v1(
            lse,
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            tp_rank,
            blocksparse_local_blocks,
            blocksparse_vert_stride,
            blocksparse_block_size,
            blocksparse_head_sliding_step,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        tmp_lse = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        ops.paged_attention_v2(
            lse,
            output,
            tmp_lse,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            tp_rank,
            blocksparse_local_blocks,
            blocksparse_vert_stride,
            blocksparse_block_size,
            blocksparse_head_sliding_step,
        )
        # TODO(ray): check the case that max_num_partitions > 1
        # max_lse, _ = tmp_lse.max(dim=-1, keepdim=True) # (num_reqs, num_heads, 1)
        # se = (tmp_lse - max_lse).exp().sum(-1) # (num_seqs, num_heads)
        # lse_ = se.log() + max_lse.squeeze(-1)
        # print(lse)
        # print(lse_)
    return output, lse
