import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from .mha import MultiHeadAttention


import math
import warnings
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union


from einops import rearrange

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t, scale = 1.):
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    

class MultiheadDiffAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_diffattn = nn.MultiheadDiffAttention(embed_dim, num_heads, depth)
        >>> attn_output, attn_output_weights = multihead_diffattn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, depth, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadDiffAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.depth = depth
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads // 2
        assert self.head_dim * num_heads *2 == self.embed_dim, "embed_dim must be divisible by num_heads"

         
        
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        #DiffAttention
        self.lambda_init = self.lambda_init_fn()
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.rms_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
        # self.rms_norm = nn.LayerNorm(2*self.head_dim)
        self._reset_parameters()
    


    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadDiffAttention, self).__setstate__(state)
    
    def lambda_init_fn(self):
        return 0.8 - 0.6 * math.exp(-0.3 * self.depth)

    
    
    def _mha_shape_check(self, query: Tensor, key: Tensor, value: Tensor,
                        key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor]):
        # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
        # and returns if the input is batched or not.
        # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

        # Shape check.
        if query.dim() == 3:
            # Batched Inputs
            is_batched = True
            assert key.dim() == 3 and value.dim() == 3, \
                ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
                f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
            if key_padding_mask is not None:
                assert key_padding_mask.dim() == 2, \
                    ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                    f" but found {key_padding_mask.dim()}-D tensor instead")
            if attn_mask is not None:
                assert attn_mask.dim() in (2, 3), \
                    ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                    f" but found {attn_mask.dim()}-D tensor instead")
        elif query.dim() == 2:
            # Unbatched Inputs
            is_batched = False
            assert key.dim() == 2 and value.dim() == 2, \
                ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
                f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

            if key_padding_mask is not None:
                assert key_padding_mask.dim() == 1, \
                    ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                    f" but found {key_padding_mask.dim()}-D tensor instead")

            if attn_mask is not None:
                assert attn_mask.dim() in (2, 3), \
                    ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                    f" but found {attn_mask.dim()}-D tensor instead")
                if attn_mask.dim() == 3:
                    expected_shape = (self.num_heads, query.shape[0], key.shape[0])
                    assert attn_mask.shape == expected_shape, \
                        (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
        else:
            raise AssertionError(
                f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

        return is_batched
    
    def _in_projection_packed(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
    ) -> List[Tensor]:
        r"""
        Performs the in-projection step of the attention operation, using packed weights.
        Output is a triple containing projection tensors for query, key and value.

        Args:
            q, k, v: query, key and value tensors to be projected. For self-attention,
                these are typically the same tensor; for encoder-decoder attention,
                k and v are typically the same tensor. (We take advantage of these
                identities for performance if they are present.) Regardless, q, k and v
                must share a common embedding dimension; otherwise their shapes may vary.
            w: projection weights for q, k and v, packed into a single tensor. Weights
                are packed along dimension 0, in q, k, v order.
            b: optional projection biases for q, k and v, packed into a single tensor
                in q, k, v order.

        Shape:
            Inputs:
            - q: :math:`(..., E)` where E is the embedding dimension
            - k: :math:`(..., E)` where E is the embedding dimension
            - v: :math:`(..., E)` where E is the embedding dimension
            - w: :math:`(E * 3, E)` where E is the embedding dimension
            - b: :math:`E * 3` where E is the embedding dimension

            Output:
            - in output list :math:`[q', k', v']`, each output tensor will have the
                same shape as the corresponding input tensor.
        """
        E = q.size(-1)
        if k is v:
            if q is k:
                # self-attention
                return F.linear(q, w, b).chunk(3, dim=-1)
            else:
                # encoder-decoder attention
                w_q, w_kv = w.split([E, E * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([E, E * 2])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


    def _in_projection(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        b_q: Optional[Tensor] = None,
        b_k: Optional[Tensor] = None,
        b_v: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Performs the in-projection step of the attention operation. This is simply
        a triple of linear projections, with shape constraints on the weights which
        ensure embedding dimension uniformity in the projected outputs.
        Output is a triple containing projection tensors for query, key and value.

        Args:
            q, k, v: query, key and value tensors to be projected.
            w_q, w_k, w_v: weights for q, k and v, respectively.
            b_q, b_k, b_v: optional biases for q, k and v, respectively.

        Shape:
            Inputs:
            - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
                number of leading dimensions.
            - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
                number of leading dimensions.
            - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
                number of leading dimensions.
            - w_q: :math:`(Eq, Eq)`
            - w_k: :math:`(Eq, Ek)`
            - w_v: :math:`(Eq, Ev)`
            - b_q: :math:`(Eq)`
            - b_k: :math:`(Eq)`
            - b_v: :math:`(Eq)`

            Output: in output triple :math:`(q', k', v')`,
            - q': :math:`[Qdims..., Eq]`
            - k': :math:`[Kdims..., Eq]`
            - v': :math:`[Vdims..., Eq]`

        """
        Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
        assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
        assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
        assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
        assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
        assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
        assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


    

    
    def multi_head_diff_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p: float,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        pos = None,
        query_pos = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
            add_zero_attn: add a new batch of zeros to the key and
                        value sequences at dim=1.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            use_separate_proj_weight: the function accept the proj. weights for query, key,
                and value in different forms. If false, in_proj_weight will be used, which is
                a combination of q_proj_weight, k_proj_weight, v_proj_weight.
            q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
            static_k, static_v: static key and value used for attention operators.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
                Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
                when ``need_weights=True.``. Default: True


        Shape:
            Inputs:
            - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
            N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
            - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
            N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

            Outputs:
            - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
            attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
            :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
            :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
            head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
        """
        tens_ops = (query, key, value, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.out_proj.weight, self.out_proj.bias)
        if has_torch_function(tens_ops):
            return handle_torch_function(
                self.multi_head_diff_attention_forward,
                tens_ops,
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.out_proj.weight,
                self.out_proj.bias,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                static_k=static_k,
                static_v=static_v,
            )

        is_batched = self._mha_shape_check(query, key, value, key_padding_mask, attn_mask)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == self.embed_dim, \
            f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert key.shape[:2] == value.shape[:2], \
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        else:
            assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            q, k, v = self._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        else:
            assert self.q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            assert self.k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            assert self.v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            q, k, v = self._in_projection(query, key, value, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, b_q, b_k, b_v)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # add bias along batch dimension (currently second)
        if self.bias_k is not None and self.bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        #
        # reshape q, k, v for multihead diffattention and make them batch first
        #
        q = q.contiguous().view(tgt_len, bsz * self.num_heads * 2, self.head_dim).transpose(0, 1)
        query_pos = query_pos.contiguous().view(tgt_len, bsz * self.num_heads * 2, self.head_dim).transpose(0,1)
        if static_k is None:
            k = k.contiguous().view(k.shape[0], bsz * self.num_heads * 2, self.head_dim).transpose(0, 1)
            pos = pos.contiguous().view(pos.shape[0], bsz * self.num_heads * 2, self.head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * self.num_heads * 2, \
                f"expecting static_k.size(0) of {bsz * self.num_heads * 2}, but got {static_k.size(0)}"
            assert static_k.size(2) == self.head_dim, \
                f"expecting static_k.size(2) of {self.head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.contiguous().view(v.shape[0], bsz * self.num_heads, 2* self.head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * self.num_heads, \
                f"expecting static_v.size(0) of {bsz * self.num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == 2 * self.head_dim, \
                f"expecting static_v.size(2) of {2 * self.head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            k_zero_attn_shape = (bsz * self.num_heads * 2, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(k_zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v_zero_attn_shape = (bsz * self.num_heads, 1, 2 * self.head_dim)
            v = torch.cat([v, torch.zeros(v_zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, self.num_heads * 2, -1, -1).reshape(bsz * self.num_heads * 2, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        
        q = apply_rotary_pos_emb(query_pos, q)

        k = apply_rotary_pos_emb(pos, k)
        #
        # (deep breath) calculate attention and out projection
        #
        #Scaled Diff Dot Product
        _, _, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn_output_weights = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn_output_weights += attn_mask
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)
        
        # Diff Attention
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_output_weights = attn_output_weights[:, :, 0] - lambda_full * attn_output_weights[:, :, 1]
        
        
        attn_output = torch.matmul(attn_output_weights, v.view(bsz, self.num_heads, -1, self.head_dim * 2))
        
        attn_output = self.rms_norm(attn_output)
        attn_output = attn_output * (1 - self.lambda_init)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            return attn_output, None


    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True, pos=None, query_pos=None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True.``. Default: True (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_diff_attention_forward(
                query, key, value,
                self.dropout if self.training else 0.0,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                average_attn_weights=average_attn_weights,
                pos=pos, query_pos=query_pos)
        else:
            attn_output, attn_output_weights = self.multi_head_diff_attention_forward(
                query, key, value,
                self.dropout if self.training else 0.0,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                pos=pos, query_pos = query_pos)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

