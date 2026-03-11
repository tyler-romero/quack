# Copyright (c) 2025, Tri Dao.

import pytest
import torch
import torch.nn.functional as F

from quack.linear_cross_entropy import (
    chunked_linear_cross_entropy,
    linear_cross_entropy_func_ref,
)


def linear_cross_entropy_z_loss_ref(
    x, weight, target, z_loss_weight, ignore_index=-100, reduction="mean"
):
    """Reference implementation of linear cross entropy with z-loss."""
    logits = F.linear(x, weight)  # (B*L, V)
    ce_loss = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    valid = target != ignore_index
    lse = torch.logsumexp(logits, dim=-1)  # (B*L,)
    z_loss_per_token = lse.square()
    z_loss_per_token[~valid] = 0.0
    ce_loss[~valid] = 0.0
    total_per_token = ce_loss + z_loss_weight * z_loss_per_token
    n_valid = valid.sum().float()
    if reduction == "mean":
        total_loss = total_per_token.sum() / n_valid
        z_loss = z_loss_weight * z_loss_per_token.sum() / n_valid
    else:
        total_loss = total_per_token.sum()
        z_loss = z_loss_weight * z_loss_per_token.sum()
    return total_loss, z_loss


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("V", [32000, 50264, 128256])
# @pytest.mark.parametrize("V", [32000])
@pytest.mark.parametrize("d", [768, 1024])
# @pytest.mark.parametrize("d", [768])
@pytest.mark.parametrize("B_L", [8, 16, 24])
@pytest.mark.parametrize("chunk_size", [16])
def test_chunked_linear_cross_entropy(B_L, d, V, chunk_size, reduction, input_dtype):
    """Test chunked linear cross entropy against reference implementation."""
    device = "cuda"
    atol, rtol = 1e-3, 1e-3
    torch.random.manual_seed(0)
    x = (torch.randn(B_L, d, device=device, dtype=input_dtype) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=input_dtype) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)
    loss_ref = linear_cross_entropy_func_ref(
        x_ref.float(), weight_ref.float(), None, target, reduction=reduction
    )
    loss_pt = linear_cross_entropy_func_ref(x_pt, weight_pt, None, target, reduction=reduction)
    # Chunked implementation
    loss = chunked_linear_cross_entropy(
        x, weight, target, chunk_size=chunk_size, reduction=reduction, tuned=False
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("chunk_size", [256, 1024])
def test_chunked_linear_cross_entropy_ignore_index(input_dtype, reduction, chunk_size):
    """Test chunked linear cross entropy with ignore_index."""
    device = "cuda"
    B_L, d, V = 1024, 512, 2048
    ignore_index = V - 1
    atol, rtol = 1e-3, 1e-3
    torch.random.manual_seed(42)
    x = (torch.randn(B_L, d, device=device, dtype=input_dtype) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=input_dtype) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)
    # Set some targets to ignore_index
    ignore_mask = torch.rand(B_L, device=device) < 0.2
    target[ignore_mask] = ignore_index
    loss_ref = linear_cross_entropy_func_ref(
        x_ref.float(), weight_ref.float(), None, target, reduction=reduction
    )
    loss_pt = linear_cross_entropy_func_ref(x_pt, weight_pt, None, target, reduction=reduction)
    # Chunked implementation
    loss = chunked_linear_cross_entropy(
        x, weight, target, chunk_size=chunk_size, reduction=reduction, tuned=False
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("V", [32000, 128256])
@pytest.mark.parametrize("d", [768])
@pytest.mark.parametrize("B_L", [8, 24])
@pytest.mark.parametrize("chunk_size", [16])
@pytest.mark.parametrize("z_loss_weight", [1e-4, 1.0])
def test_chunked_linear_cross_entropy_z_loss(
    B_L, d, V, chunk_size, reduction, input_dtype, z_loss_weight
):
    """Test chunked linear cross entropy with z-loss against reference."""
    device = "cuda"
    torch.random.manual_seed(0)
    x = (torch.randn(B_L, d, device=device, dtype=input_dtype) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=input_dtype) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().float().requires_grad_(True)
    weight_ref = weight.detach().clone().float().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)
    # Reference (float32)
    loss_ref, z_loss_ref = linear_cross_entropy_z_loss_ref(
        x_ref, weight_ref, target, z_loss_weight, reduction=reduction
    )
    # Reference (input dtype)
    loss_pt, z_loss_pt = linear_cross_entropy_z_loss_ref(
        x_pt, weight_pt, target, z_loss_weight, reduction=reduction
    )
    # Chunked implementation
    loss, z_loss = chunked_linear_cross_entropy(
        x,
        weight,
        target,
        chunk_size=chunk_size,
        reduction=reduction,
        tuned=False,
        z_loss_weight=z_loss_weight,
        return_z_loss=True,
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    assert (z_loss - z_loss_ref).abs().max() < 3 * (z_loss_pt - z_loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("chunk_size", [256, 1024])
@pytest.mark.parametrize("z_loss_weight", [1e-4])
def test_chunked_linear_cross_entropy_z_loss_ignore_index(
    input_dtype, reduction, chunk_size, z_loss_weight
):
    """Test chunked linear cross entropy z-loss with ignore_index."""
    device = "cuda"
    B_L, d, V = 1024, 512, 2048
    ignore_index = V - 1
    torch.random.manual_seed(42)
    x = (torch.randn(B_L, d, device=device, dtype=input_dtype) * 0.1).requires_grad_()
    weight = (torch.randn(V, d, device=device, dtype=input_dtype) / (d**0.5)).requires_grad_()
    target = torch.randint(0, V, (B_L,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().float().requires_grad_(True)
    weight_ref = weight.detach().clone().float().requires_grad_(True)
    x_pt = x.detach().clone().requires_grad_(True)
    weight_pt = weight.detach().clone().requires_grad_(True)
    # Set some targets to ignore_index
    ignore_mask = torch.rand(B_L, device=device) < 0.2
    target[ignore_mask] = ignore_index
    # Reference (float32)
    loss_ref, z_loss_ref = linear_cross_entropy_z_loss_ref(
        x_ref, weight_ref, target, z_loss_weight, ignore_index=ignore_index, reduction=reduction
    )
    # Reference (input dtype)
    loss_pt, z_loss_pt = linear_cross_entropy_z_loss_ref(
        x_pt, weight_pt, target, z_loss_weight, ignore_index=ignore_index, reduction=reduction
    )
    # Chunked implementation
    loss, z_loss = chunked_linear_cross_entropy(
        x,
        weight,
        target,
        chunk_size=chunk_size,
        ignore_index=ignore_index,
        reduction=reduction,
        tuned=False,
        z_loss_weight=z_loss_weight,
        return_z_loss=True,
    )
    assert (loss - loss_ref).abs().max() < 3 * (loss_pt - loss_ref).abs().max() + 1e-5
    assert (z_loss - z_loss_ref).abs().max() < 3 * (z_loss_pt - z_loss_ref).abs().max() + 1e-5
    loss.backward()
    loss_ref.backward()
    loss_pt.backward()
    assert (x.grad - x_ref.grad).abs().max() < 2 * (x_pt.grad - x_ref.grad).abs().max() + 1e-4
    assert (weight.grad - weight_ref.grad).abs().max() < 2 * (
        weight_pt.grad - weight_ref.grad
    ).abs().max() + 1e-4
