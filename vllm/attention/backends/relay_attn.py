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
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.backends.functional.flashattn_ops import flash_attn_with_kvcache
from vllm.attention.backends.functional.relayattn_ops import relay_fusion
from vllm.attention.backends.functional.xformers_ops import memory_efficient_attention_forward

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

    prefix_length: int
    prefix_length_buffer: Optional[torch.Tensor]

    # slot_mapping: torch.Tensor is in abstract class

    attn_bias: Optional[List[AttentionBias]]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # The data bellow used only for CommonMetadataBuilder

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int]

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    _cached_prefill_metadata: Optional["RelayAttentionMetadata"] = None
    _cached_decode_metadata: Optional["RelayAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["RelayAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None

        self._cached_prefill_metadata = RelayAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            seq_lens=self.seq_lens,
            prefix_length=0,
            prefix_length_buffer=None,
            slot_mapping=self.slot_mapping,
            attn_bias=self.attn_bias,
            block_tables=None,
            seq_lens_tensor=None,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
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

        seq_lens_tensor = self.seq_lens_tensor[self.num_prefills:]
        block_tables = self.block_tables[self.num_prefills:]

        self._cached_prefill_metadata = RelayAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            seq_lens=self.seq_lens,
            prefix_length=0,
            prefix_length_buffer=None,
            slot_mapping=self.slot_mapping,
            attn_bias=None,
            block_tables=block_tables,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
        )
        return self._cached_decode_metadata

class RelayAttentionMetadataBuilder(
        CommonMetadataBuilder[RelayAttentionMetadata]):
    _metadata_cls = RelayAttentionMetadata


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

        if prefill_meta := attn_metadata.prefill_metadata:
            # fill the prefix kv cache here
            # NOTE: this happens only when we fill prefix kv cache
            # by passing input_metadata.prefix_length=-1 as the signal
            if prefill_meta.prefix_length < 0:
                assert (kv_cache is None) and (prefix_kv_cache is not None)
                assert (len(prefill_meta.seq_lens) == 1)
                seq_len = prefill_meta.seq_lens[0]
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

        if prefill_meta := attn_metadata.prefill_metadata:
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
            if prefill_meta.attn_bias is None:
                attn_bias = BlockDiagonalCausalMask.from_seqlens(
                    prefill_meta.seq_lens)
                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                prefill_meta.attn_bias = attn_bias

            # [1, bsz*seqlen, head_groups, num_heads_per_group, K]
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

            out, lse = memory_efficient_attention_forward(query,
                                                          key,
                                                          value,
                                                          attn_bias=prefill_meta.attn_bias,
                                                          p=0.0,
                                                          scale=self.scale)
            output = out.view(num_tokens, self.num_heads, self.head_size)
            # (bsz, head, num_queries) -> (bsz*num_queries, nheads)
            lse = lse.transpose(1, 2).reshape(num_tokens, self.num_heads).contiguous()

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            if kv_cache is not None:
                # (bsz*seqlen, nheads, head_size)
                # (bsz*seqlen, nheads)
                output, lse = _paged_attention(
                    query,
                    kv_cache[0],
                    kv_cache[1],
                    decode_meta.block_tables,
                    decode_meta.seq_lens_tensor,
                    decode_meta.max_decode_seq_len,
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
