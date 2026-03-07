# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import math
from functools import partial
from typing import Optional, Type, Literal

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr

import quack.utils as utils
import quack.copy_utils as copy_utils
import quack.layout_utils as layout_utils
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.reduce import row_reduce
from quack.reduction_base import ReductionBase
from quack.cute_dsl_utils import torch2cute_dtype_map


class CrossEntropyChunkedFwd(ReductionBase):
    """Cross entropy that computes dlogits in the forward pass using a chunked 2-pass approach.

    Similar to Liger kernel's cross entropy: iterates over the vocabulary dimension in chunks,
    avoiding materializing the full softmax vector. Pass 1 computes online softmax statistics
    (max, sum_exp), pass 2 computes gradients using the final statistics.
    """

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, chunk_N: int):
        self.full_N = N
        self.chunk_N = chunk_N
        self.num_chunks = math.ceil(N / chunk_N)
        super().__init__(dtype, chunk_N, stage=1, reduction_dtype=Float32)

    def _threads_per_row(self):
        N = self.chunk_N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mLoss: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mdX: cute.Tensor,
        ignore_index: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mdX.element_type == self.dtype
        self.cluster_n = 1
        largest_dtype_width = const_expr(max(mX.element_type.width, mdX.element_type.width))
        vecsize = math.gcd(self.chunk_N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(
            mX,
            mTarget,
            mLoss,
            mLSE,
            mdX,
            ignore_index,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mLoss: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mdX: cute.Tensor,
        ignore_index: Int32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)
        tXsX = thr_copy.partition_D(sX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        # Get row index from the first chunk's coordinate tile
        gX_0 = cute.local_tile(mX, tiler_mn, (bidx, 0))
        cX_0 = cute.local_tile(idX, tiler_mn, (bidx, 0))
        tXcX_0 = thr_copy.partition_S(cX_0)[(0, None), None, None]
        row = tXcX_0[0][0]

        target = Int32.zero
        should_ignore = Boolean(True)
        if row < shape[0]:
            target = Int32(mTarget[row])
            should_ignore = Boolean(target == ignore_index)

        # Load target logit
        target_logit = Float32.zero
        if row < shape[0] and tXcX_0[0][1] == 0 and not should_ignore:
            target_logit = Float32(mX[row, target])

        # ---- Pass 1: Online softmax over chunks to compute max and sum_exp ----
        max_val = -Float32.inf
        sum_exp = Float32.zero
        num_chunks = const_expr(self.num_chunks)
        is_even_N = const_expr(self.full_N % self.chunk_N == 0)

        for chunk_idx in cutlass.range(num_chunks, unroll_full=const_expr(num_chunks <= 4)):
            gX_chunk = cute.local_tile(mX, tiler_mn, (bidx, chunk_idx))
            cX_chunk = cute.local_tile(idX, tiler_mn, (bidx, chunk_idx))
            tXgX_chunk = thr_copy.partition_S(gX_chunk)
            tXrX_chunk = cute.make_fragment_like(tXgX_chunk)

            if const_expr(not is_even_N):
                tXpX_chunk = copy_utils.predicate_k(
                    thr_copy.partition_S(cX_chunk), limit=shape[1]
                )
                copy_chunk = partial(copy_utils.copy, pred=tXpX_chunk)
            else:
                copy_chunk = copy_utils.copy

            if row < shape[0]:
                copy_chunk(tXgX_chunk, tXsX, is_async=True)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            if const_expr(not is_even_N):
                utils.fill_oob(tXsX, tXpX_chunk, -tXsX.element_type.inf)
            cute.autovec_copy(tXsX, tXrX_chunk)
            x = tXrX_chunk.load().to(Float32)

            chunk_max = row_reduce(
                x,
                cute.ReductionOp.MAX,
                threads_per_row,
                reduction_buffer[None, None, 0],
                None,
                init_val=-Float32.inf,
            )
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2(x * log2_e - (chunk_max * log2_e), fastmath=False)
            chunk_sum_exp = row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                None,
                init_val=0.0,
            )

            # Online softmax accumulation across chunks
            new_max = cute.arch.fmax(max_val, chunk_max)
            sum_exp = (
                sum_exp * cute.math.exp(max_val - new_max, fastmath=True)
                + chunk_sum_exp * cute.math.exp(chunk_max - new_max, fastmath=True)
            )
            max_val = new_max

        # Compute loss and lse
        lse = max_val + cute.math.log(sum_exp, fastmath=True)
        if (
            tXcX_0[0][1] == 0
            and row < shape[0]
        ):
            loss_val = (lse - target_logit) if not should_ignore else Float32.zero
            mLoss[row] = mLoss.element_type(loss_val)
            if const_expr(mLSE is not None):
                mLSE[row] = lse

        # ---- Pass 2: Compute gradients using final lse ----
        denom_inv = (
            cute.arch.rcp_approx(sum_exp)
            if not (sum_exp == 0.0 or sum_exp != sum_exp or should_ignore)
            else Float32.zero
        )

        for chunk_idx in cutlass.range(num_chunks, unroll_full=const_expr(num_chunks <= 4)):
            gX_chunk = cute.local_tile(mX, tiler_mn, (bidx, chunk_idx))
            gdX_chunk = cute.local_tile(mdX, tiler_mn, (bidx, chunk_idx))
            cX_chunk = cute.local_tile(idX, tiler_mn, (bidx, chunk_idx))
            tXgX_chunk = thr_copy.partition_S(gX_chunk)
            tXgdX_chunk = thr_copy.partition_D(gdX_chunk)
            tXrX_chunk = cute.make_fragment_like(tXgX_chunk)
            tXrdX_chunk = cute.make_fragment_like(tXgdX_chunk)
            tXcFull_chunk = thr_copy.partition_S(cX_chunk)

            if const_expr(not is_even_N):
                tXpX_chunk = copy_utils.predicate_k(
                    thr_copy.partition_S(cX_chunk), limit=shape[1]
                )
                copy_chunk = partial(copy_utils.copy, pred=tXpX_chunk)
            else:
                copy_chunk = copy_utils.copy

            if row < shape[0]:
                copy_chunk(tXgX_chunk, tXsX, is_async=True)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            if const_expr(not is_even_N):
                utils.fill_oob(tXsX, tXpX_chunk, -tXsX.element_type.inf)
            cute.autovec_copy(tXsX, tXrX_chunk)
            x = tXrX_chunk.load().to(Float32)

            log2_e = math.log2(math.e)
            probs = cute.math.exp2(x * log2_e - (max_val * log2_e), fastmath=True) * denom_inv

            tXrdX_f32 = cute.make_fragment_like(tXrX_chunk, Float32)
            tXrdX_f32.store(probs)
            if not should_ignore:
                for i in cutlass.range(cute.size(tXrX_chunk), unroll_full=True):
                    tXrdX_f32[i] = (
                        tXrdX_f32[i] if tXcFull_chunk[i][1] != target else tXrdX_f32[i] - 1.0
                    )
            tXrdX_chunk.store(tXrdX_f32.load().to(tXrdX_chunk.element_type))
            if row < shape[0]:
                copy_chunk(tXrdX_chunk, tXgdX_chunk)


def _default_chunk_n(N: int) -> int:
    if N <= 4096:
        return N
    elif N <= 32768:
        return 4096
    else:
        return 8192


@torch.library.custom_op(
    "quack::cross_entropy_chunked_fwd_out", mutates_args={"loss", "lse", "dx"}
)
def cross_entropy_chunked_fwd_out(
    x: Tensor,
    target: Tensor,
    loss: Tensor,
    lse: Optional[Tensor],
    dx: Tensor,
    ignore_index: int = -100,
    chunk_n: int = 0,
) -> None:
    assert x.dim() == 2, "Input must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert x.is_cuda and target.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert target.dtype in [torch.int32, torch.int64], "Target must be int32 or int64"
    assert dx.is_cuda, "dx must be on CUDA device"
    N = x.size(1)
    chunk_n = chunk_n if chunk_n > 0 else _default_chunk_n(N)
    dtype = torch2cute_dtype_map[x.dtype]
    target_dtype = torch2cute_dtype_map[target.dtype]
    compile_key = (dtype, target_dtype, N, lse is not None, chunk_n)
    if compile_key not in cross_entropy_chunked_fwd_out.compile_cache:
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        dx_cute = fake_tensor(dtype, (batch_sym, N), div)
        target_cute = fake_tensor(target_dtype, (batch_sym,))
        loss_cute = fake_tensor(Float32, (batch_sym,))
        lse_cute = fake_tensor(Float32, (batch_sym,)) if lse is not None else None
        op = CrossEntropyChunkedFwd(dtype, N, chunk_n)
        cross_entropy_chunked_fwd_out.compile_cache[compile_key] = cute.compile(
            op,
            x_cute,
            target_cute,
            loss_cute,
            lse_cute,
            dx_cute,
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    cross_entropy_chunked_fwd_out.compile_cache[compile_key](
        x, target, loss, lse, dx, Int32(ignore_index)
    )


cross_entropy_chunked_fwd_out.compile_cache = {}


def cross_entropy_chunked_fwd(
    x: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    return_lse: bool = False,
    inplace_backward: bool = False,
    chunk_n: int = 0,
) -> torch.Tensor | tuple[torch.Tensor]:
    M = x.size(0)
    device = x.device
    loss = torch.empty(M, device=device, dtype=torch.float32)
    lse = torch.empty(M, device=device, dtype=torch.float32) if return_lse else None
    dx = x if inplace_backward else torch.empty_like(x)
    cross_entropy_chunked_fwd_out(x, target, loss, lse, dx, ignore_index, chunk_n)
    if return_lse:
        return loss, lse, dx
    return loss, dx


class CrossEntropyChunkedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, ignore_index=-100, inplace_backward=False, chunk_n=0):
        loss, lse, dx = cross_entropy_chunked_fwd(
            x, target, ignore_index=ignore_index, return_lse=True,
            inplace_backward=inplace_backward, chunk_n=chunk_n,
        )
        ctx.save_for_backward(dx, lse)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, dloss):
        dx, lse = ctx.saved_tensors
        # dx already contains (softmax - one_hot), just scale by dloss
        dx = dx * dloss.unsqueeze(1)
        return dx, None, None, None, None


def cross_entropy_chunked(
    x: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
    inplace_backward: bool = False,
    chunk_n: int = 0,
) -> torch.Tensor:
    """Cross entropy loss that computes dlogits in the forward pass using chunked 2-pass approach.

    Similar to Liger kernel's cross entropy: processes the vocabulary dimension in chunks,
    computing online softmax statistics in pass 1 and gradients in pass 2. This avoids
    materializing the full softmax vector and doesn't require Hopper cluster support.

    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)
        ignore_index: Index to ignore in loss computation
        reduction: 'none', 'mean', or 'sum'
        inplace_backward: Whether to write gradients into x in-place
        chunk_n: Chunk size along vocab dimension (0 = auto)

    Returns:
        Cross entropy loss tensor
    """
    loss = CrossEntropyChunkedFunction.apply(x, target, ignore_index, inplace_backward, chunk_n)
    if reduction == "mean":
        return loss.sum() / (target != ignore_index).sum().float()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', or 'sum'"
        )
