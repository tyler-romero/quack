# Copyright (c) 2025, Tri Dao
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_fwd, custom_bwd

from quack.cross_entropy import cross_entropy, cross_entropy_fwd_out
from quack.gemm_interface import gemm, gemm_add, gemm_add_inplace
from quack.linear import linear_fwd_convert_type


def linear_cross_entropy_func(
    x: Tensor,  # (..., d)
    weight: Tensor,  # (V, d)
    bias: Optional[Tensor],  # (V,) or None
    target: Tensor,  # (...,), int or long
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
    inplace_backward: bool = False,
) -> Tensor:
    y = F.linear(x, weight, bias)  # (..., V)
    return cross_entropy(
        y, target, ignore_index=ignore_index, reduction=reduction, inplace_backward=inplace_backward
    )


def linear_cross_entropy_func_ref(
    x: Tensor,  # (..., d)
    weight: Tensor,  # (V, d)
    bias: Optional[Tensor],  # (V,) or None
    target: Tensor,  # (...,), int or long
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    y = F.linear(x, weight, bias)  # (..., V)
    return F.cross_entropy(y, target, ignore_index=ignore_index, reduction=reduction)


def chunked_linear_cross_entropy_fwd(
    x: Tensor,  # (B*L, d) where B is batch, L is seqlen
    weight: Tensor,  # (V, d) where V is vocab size
    target: Tensor,  # (B*L,)
    chunk_size: int = 4096,
    ignore_index: int = -100,
    tuned: bool = True,
    z_loss_weight: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Chunked forward pass for linear cross entropy.

    Splits input along batch dimension, computes matmul and cross_entropy_fwd
    for each chunk, stores dx for each chunk, and accumulates dw.

    Returns:
        loss: (B*L,) loss values (includes z-loss contribution when z_loss_weight > 0)
        dx: (B*L, d) gradient w.r.t. input
        dw: (V, d) gradient w.r.t. weight (accumulated across chunks except last)
        last_dlogits_chunk: (chunk_len, V) gradient of last chunk's logits (for deferred dw computation)
        last_x_chunk: (chunk_len, d) last chunk's input (for deferred dw computation)
        z_loss_sum: scalar sum of lse^2 over valid tokens (None when z_loss_weight == 0)
    """
    B_L, d = x.shape
    V, _ = weight.shape
    device = x.device
    num_chunks = (B_L + chunk_size - 1) // chunk_size
    # Since we use gemm with TMA we require some alignment
    assert chunk_size % 8 == 0, "chunk_size must be multiple of 8"
    assert B_L % 8 == 0
    has_z_loss = z_loss_weight > 0.0
    # Pre-allocate outputs
    loss = torch.empty(B_L, device=device, dtype=torch.float32)
    logits_chunk_preallocated = torch.empty((chunk_size, V), device=device, dtype=x.dtype)
    dx = torch.empty_like(x)
    lse = torch.empty(B_L, device=device, dtype=torch.float32) if has_z_loss else None
    # Last chunk of dw will be deferred to the backward pass
    dw = torch.empty_like(weight, dtype=torch.float32) if num_chunks > 1 else None
    last_dlogits_chunk = None
    last_x_chunk = None

    # Process in chunks
    for i, (x_chunk, target_chunk, loss_chunk, dx_chunk) in enumerate(
        zip(*(t.split(chunk_size) for t in (x, target, loss, dx)))
    ):
        chunk_len = x_chunk.shape[0]
        chunk_start = i * chunk_size
        lse_chunk = lse[chunk_start : chunk_start + chunk_len] if has_z_loss else None
        logits_chunk = logits_chunk_preallocated[:chunk_len]  # (chunk_len, V)
        torch.mm(x_chunk, weight.mT, out=logits_chunk)
        # Compute cross entropy forward with gradients
        dlogits_chunk = logits_chunk  # inplace_backward
        cross_entropy_fwd_out(
            logits_chunk,
            target_chunk,
            None,  # target_logit
            loss=loss_chunk,
            lse=lse_chunk,
            dx=dlogits_chunk,
            ignore_index=ignore_index,
        )
        # Fix up dlogits for z-loss: add 2 * z_loss_weight * lse * softmax(logits) contribution
        # dlogits currently = probs - one_hot(target)
        # desired = (1 + 2*z*lse) * probs - one_hot(target)
        # so: scale by (1 + 2*z*lse), then add 2*z*lse at target positions
        if has_z_loss:
            z_scale = (2.0 * z_loss_weight * lse_chunk).to(dlogits_chunk.dtype).unsqueeze(1)
            dlogits_chunk *= 1.0 + z_scale
            valid = target_chunk != ignore_index
            valid_idx = valid.nonzero(as_tuple=True)[0]
            if valid_idx.numel() > 0:
                dlogits_chunk[valid_idx, target_chunk[valid_idx]] += z_scale[valid_idx, 0]
        # Compute dx for this chunk: dlogits @ weight
        torch.mm(dlogits_chunk, weight, out=dx_chunk)  # (chunk_len, d)
        # Compute dw for all chunks except the last
        if i == num_chunks - 1:
            # Last chunk: save for backward pass
            last_dlogits_chunk = dlogits_chunk
            last_x_chunk = x_chunk
        elif i == 0:
            # First chunk: dw = dlogits.T @ x_chunk
            gemm(dlogits_chunk.T, x_chunk, out=dw, tuned=tuned)
        else:
            # Middle chunks: dw += dlogits.T @ x_chunk
            gemm_add_inplace(dlogits_chunk.T, x_chunk, dw, tuned=tuned)

    # Compute z-loss and add to per-token loss
    z_loss_sum = None
    if has_z_loss:
        valid = target != ignore_index
        lse_sq = lse.square()
        lse_sq[~valid] = 0.0
        z_loss_sum = z_loss_weight * lse_sq.sum()
        loss += z_loss_weight * lse_sq

    return loss, dx, dw, last_dlogits_chunk, last_x_chunk, z_loss_sum


class ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        target: Tensor,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum"] = "mean",
        chunk_size: int = 4096,
        tuned: bool = True,
        z_loss_weight: float = 0.0,
    ):
        """
        Forward pass computes loss and stores dx and dw for backward.
        """
        ctx.weight_dtype = weight.dtype
        x, weight = linear_fwd_convert_type(x, weight)
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        # TODO: don't need to compute bwd if neither x nor weight requires grad, or not training
        loss, dx, dw, last_dlogits_chunk, last_x_chunk, z_loss_sum = (
            chunked_linear_cross_entropy_fwd(
                x,
                weight,
                target,
                chunk_size,
                ignore_index,
                tuned=tuned,
                z_loss_weight=z_loss_weight,
            )
        )
        loss_sum = loss.sum()
        loss_scale = None if reduction == "sum" else 1.0 / (target != ignore_index).sum().float()
        ctx.save_for_backward(dx, dw, last_dlogits_chunk, last_x_chunk, loss_scale)
        ctx.batch_shape = batch_shape
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.tuned = tuned
        total_loss = loss_sum if loss_scale is None else loss_sum * loss_scale
        # z_loss for logging (detached, does not affect backward)
        if z_loss_sum is not None:
            z_loss_out = z_loss_sum if loss_scale is None else z_loss_sum * loss_scale
        else:
            z_loss_out = total_loss.new_zeros(())
        return total_loss, z_loss_out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dloss, _dz_loss):
        """
        Backward pass scales pre-computed gradients by dloss and completes
        the last chunk's dw computation.
        dloss is a scalar. _dz_loss is ignored (z_loss output is for logging only).
        """
        dx, dw, last_dlogits_chunk, last_x_chunk, loss_scale = ctx.saved_tensors
        tuned = ctx.tuned
        if loss_scale is not None:
            dloss = dloss * loss_scale
        # TODO: the case where x or weight doesn't require grad
        dx.mul_(dloss)
        dx = dx.reshape(*ctx.batch_shape, dx.shape[-1])
        # Complete dw computation: dw = dloss * dw + dloss * (last_dlogits_chunk.T @ last_x_chunk)
        if dw is None:
            # Only had one chunk, compute dw directly with dloss scaling
            dw = gemm(
                last_dlogits_chunk.T,
                last_x_chunk,
                out_dtype=ctx.weight_dtype,
                alpha=dloss,
                tuned=tuned,
            )
        else:
            # Add last chunk's contribution with dloss scaling
            # dw = dloss * dw + dloss * (last_dlogits_chunk.T @ last_x_chunk)
            # We use alpha=dloss, beta=dloss
            if ctx.weight_dtype == dw.dtype:
                gemm_add_inplace(
                    last_dlogits_chunk.T, last_x_chunk, dw, alpha=dloss, beta=dloss, tuned=tuned
                )
            else:
                dw = gemm_add(
                    last_dlogits_chunk.T,
                    last_x_chunk,
                    dw,
                    alpha=dloss,
                    beta=dloss,
                    out_dtype=ctx.weight_dtype,
                    tuned=tuned,
                )
        return dx, dw, None, None, None, None, None, None


def chunked_linear_cross_entropy(
    x: Tensor,
    weight: Tensor,
    target: Tensor,
    chunk_size: int = 4096,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum"] = "mean",
    tuned: bool = True,
    z_loss_weight: float = 0.0,
    return_z_loss: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Chunked linear cross entropy with automatic differentiation support.

    Args:
        x: Input tensor of shape (B*L, d)
        weight: Weight tensor of shape (V, d)
        target: Target indices of shape (B*L,)
        chunk_size: Size of chunks to process
        ignore_index: Index to ignore in loss computation
        reduction: Type of reduction to apply
        tuned: Whether to use tuned kernels
        z_loss_weight: Weight for z-loss (lse^2 penalty). 0 disables z-loss.
        return_z_loss: If True, return (loss, z_loss) tuple where z_loss is detached.

    Returns:
        Loss tensor (includes z-loss when z_loss_weight > 0), or (loss, z_loss) tuple.
    """
    if reduction not in ["mean", "sum"]:
        raise ValueError(f"Invalid reduction: {reduction}")
    loss, z_loss = ChunkedLinearCrossEntropyFunction.apply(
        x, weight, target, ignore_index, reduction, chunk_size, tuned, z_loss_weight
    )
    if return_z_loss:
        return loss, z_loss
    return loss


class LinearCrossEntropy(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        ignore_index: int = -100,
        reduction: Literal["none", "mean", "sum"] = "mean",
        chunk_size: Optional[int] = None,
        inplace_backward: bool = False,
        tuned: bool = True,
        z_loss_weight: float = 0.0,
        return_z_loss: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.chunk_size = chunk_size
        self.inplace_backward = inplace_backward
        self.tuned = tuned
        self.z_loss_weight = z_loss_weight
        self.return_z_loss = return_z_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        if (
            self.bias is None
            and input.is_cuda
            and input.stride(-1) == 1
            and self.in_features % 8 == 0
            and self.out_features % 8 == 0
            and input.shape[:-1].numel() % 8 == 0
            and self.chunk_size is not None
            and self.chunk_size % 8 == 0
            and self.reduction in ["mean", "sum"]
        ):
            return chunked_linear_cross_entropy(
                input,
                self.weight,
                target,
                chunk_size=self.chunk_size,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                tuned=self.tuned,
                z_loss_weight=self.z_loss_weight,
                return_z_loss=self.return_z_loss,
            )
        else:
            return linear_cross_entropy_func(
                input,
                self.weight,
                self.bias,
                target,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                inplace_backward=self.inplace_backward,
            )
