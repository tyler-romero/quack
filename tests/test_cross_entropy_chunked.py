# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import pytest
import torch
import torch.nn.functional as F

from quack.cross_entropy_chunked import cross_entropy_chunked_fwd, cross_entropy_chunked

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "N",
    [192, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 128256, 131072],
)
@pytest.mark.parametrize("M", [1, 77, 289])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_chunked(M, N, input_dtype, use_compile):
    device = "cuda"
    atol, rtol = 5e-5, 1e-5
    torch.random.manual_seed(0)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().requires_grad_()
    target_ref = target.detach().clone()
    function = (
        torch.compile(cross_entropy_chunked, fullgraph=True) if use_compile else cross_entropy_chunked
    )
    loss = function(x, target, reduction="none")
    loss_ref = F.cross_entropy(x_ref.float(), target_ref, reduction="none")
    assert loss.shape == (M,)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
    assert (loss >= 0).all()
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    # Test backward pass
    dloss = torch.randn_like(loss)
    torch.cuda.synchronize()
    (dx,) = torch.autograd.grad(loss, x, grad_outputs=dloss)
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    assert dx.shape == x.shape
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_chunked_extreme_values(input_dtype, use_compile):
    device = "cuda"
    M, N = 16, 1024
    function = (
        torch.compile(cross_entropy_chunked_fwd, fullgraph=True)
        if use_compile
        else cross_entropy_chunked_fwd
    )
    # Test with large positive values
    x_large = torch.full((M, N), 10.0, device=device, dtype=input_dtype)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    loss_large, _ = function(x_large, target)
    expected_large = torch.full_like(loss_large, torch.log(torch.tensor(N, dtype=torch.float32)))
    torch.testing.assert_close(loss_large, expected_large, atol=1e-2, rtol=1e-2)
    # Test with large negative values
    x_small = torch.full((M, N), -10.0, device=device, dtype=input_dtype)
    loss_small, _ = function(x_small, target)
    torch.testing.assert_close(loss_small, expected_large, atol=1e-2, rtol=1e-2)
    # Test with one-hot like scenario
    x_onehot = torch.full((M, N), -10.0, device=device, dtype=input_dtype)
    for i in range(M):
        x_onehot[i, target[i]] = 10.0
    loss_onehot, _ = function(x_onehot, target)
    assert (loss_onehot < 1.0).all()


@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_chunked_numerical_stability(use_compile):
    device = "cuda"
    M, N = 8, 512
    function = (
        torch.compile(cross_entropy_chunked_fwd, fullgraph=True)
        if use_compile
        else cross_entropy_chunked_fwd
    )
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    x_shifted = x + 100.0
    loss, _ = function(x, target)
    loss_shifted, _ = function(x_shifted, target)
    torch.testing.assert_close(loss, loss_shifted, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [192, 1024, 32768])
@pytest.mark.parametrize("M", [1, 77, 289])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_chunked_ignore_index(M, N, input_dtype, use_compile):
    device = "cuda"
    atol, rtol = 5e-5, 1e-5
    torch.random.manual_seed(0)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    ignore_index = N - 1
    ignore_mask = torch.rand(M, device=device) < 0.3
    target[ignore_mask] = ignore_index
    x_ref = x.detach().clone().requires_grad_()
    target_ref = target.detach().clone()
    function = (
        torch.compile(cross_entropy_chunked, fullgraph=True)
        if use_compile
        else cross_entropy_chunked
    )
    loss = function(x, target, reduction="none", ignore_index=ignore_index)
    loss_ref = F.cross_entropy(
        x_ref.float(), target_ref, reduction="none", ignore_index=ignore_index
    )
    assert (loss[ignore_mask] == 0).all(), "Loss should be 0 for ignored indices"
    if (~ignore_mask).any():
        torch.testing.assert_close(loss[~ignore_mask], loss_ref[~ignore_mask], atol=atol, rtol=rtol)
    dloss = torch.randn_like(loss)
    torch.cuda.synchronize()
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    (dx,) = torch.autograd.grad(loss, x, grad_outputs=dloss)
    assert dx.shape == x.shape
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [192, 1024, 32768, 128256])
@pytest.mark.parametrize("M", [1, 77, 289])
@pytest.mark.parametrize("inplace_backward", [False, True])
@pytest.mark.parametrize("use_compile", [False, True])
def test_cross_entropy_chunked_fwd_with_grad(M, N, input_dtype, inplace_backward, use_compile):
    device = "cuda"
    atol, rtol = 1e-4, 1e-4
    torch.random.manual_seed(0)
    x = (0.1 * torch.randn(M, N, device=device, dtype=input_dtype)).requires_grad_()
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    x_ref = x.detach().clone().requires_grad_()
    target_ref = target.detach().clone()
    function = (
        torch.compile(cross_entropy_chunked_fwd, fullgraph=True)
        if use_compile
        else cross_entropy_chunked_fwd
    )
    if inplace_backward:
        x_copy = x.detach().clone()
        loss, lse, dx = function(
            x_copy, target, return_lse=True, inplace_backward=True
        )
        assert dx is x_copy, "inplace_backward should modify x in-place"
    else:
        loss, lse, dx = function(x, target, return_lse=True, inplace_backward=False)
        assert dx is not x, "non-inplace should create new tensor"
    loss_ref = F.cross_entropy(x_ref.float(), target_ref, reduction="none")
    lse_ref = torch.logsumexp(x_ref.float(), dim=-1)
    dloss = torch.ones_like(loss_ref)
    (dx_ref,) = torch.autograd.grad(loss_ref, x_ref, grad_outputs=dloss)
    torch.testing.assert_close(loss, loss_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx, dx_ref.to(input_dtype), atol=atol, rtol=rtol)
