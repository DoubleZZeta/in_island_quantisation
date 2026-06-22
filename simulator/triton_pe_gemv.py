import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import triton
import triton.language as tl
from . import quantization as q
from . import torch_pe_gemv as reference

@triton.jit
def pe_partial_gemv_kernel(
    x_ptr, W_ptr, partial_ptr, partial_scale_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    PE_COLS: tl.constexpr,
    K_PER_PE: tl.constexpr,
    N_PER_PE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_MODE: tl.constexpr,
):
    pe_row = tl.program_id(0)
    pe_col = tl.program_id(1)

    k_local = tl.arange(0, BLOCK_K)
    n_local = tl.arange(0, BLOCK_N)

    k_offsets = pe_row * K_PER_PE + k_local
    n_offsets = pe_col * N_PER_PE + n_local

    x_vals = tl.load(
        x_ptr + k_offsets,
        mask=(k_local < K_PER_PE) & (k_offsets < K),
        other=0.0,
    ).to(tl.float32)

    W_vals = tl.load(
        W_ptr + k_offsets[:, None] * N + n_offsets[None, :],
        mask=(
            (k_local[:, None] < K_PER_PE)
            & (k_offsets[:, None] < K)
            & (n_local[None, :] < N_PER_PE)
            & (n_offsets[None, :] < N)
        ),
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(x_vals[:, None] * W_vals, axis=0)

    partial_base = (pe_row * PE_COLS + pe_col) * BLOCK_N
    scale_offset = pe_row * PE_COLS + pe_col
    valid_n = (n_local < N_PER_PE) & (n_offsets < N)

    if QUANT_MODE == 3:
        q_acc = acc
    elif QUANT_MODE == 0:
        q_acc = q.quantize_fp16_tl(acc)
    elif QUANT_MODE == 1:
        scale = q.symmetric_scale_tl(acc, valid_n, 4, 1.0e-8)
        q_acc = q.quantize_symmetric_tl(acc, scale, 4)
        tl.store(partial_scale_ptr + scale_offset, scale)
    else:
        scale = q.symmetric_scale_tl(acc, valid_n, 8, 1.0e-8)
        q_acc = q.quantize_symmetric_tl(acc, scale, 8)
        tl.store(partial_scale_ptr + scale_offset, scale)

    tl.store(
        partial_ptr + partial_base + n_local,
        q_acc,
        mask=valid_n,
    )


@triton.jit
def pe_reduce_kernel(
    partial_ptr, partial_scale_ptr, y_ptr,
    N: tl.constexpr,
    PE_ROWS: tl.constexpr,
    PE_COLS: tl.constexpr,
    N_PER_PE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_MODE: tl.constexpr,
):
    pe_col = tl.program_id(0)

    n_local = tl.arange(0, BLOCK_N)
    n_offsets = pe_col * N_PER_PE + n_local

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for pe_row in range(0, PE_ROWS):
        partial_base = (pe_row * PE_COLS + pe_col) * BLOCK_N
        scale_offset = pe_row * PE_COLS + pe_col
        vals = tl.load(
            partial_ptr + partial_base + n_local,
            mask=(n_local < N_PER_PE) & (n_offsets < N),
            other=0,
        )
        if QUANT_MODE == 3:
            acc += vals
        elif QUANT_MODE == 0:
            acc += q.dequantize_fp16_tl(vals)
        else:
            scale = tl.load(partial_scale_ptr + scale_offset)
            acc += q.dequantize_symmetric_tl(vals, scale)

    tl.store(
        y_ptr + n_offsets,
        acc,
        mask=(n_local < N_PER_PE) & (n_offsets < N),
    )


def triton_pe_gemv(x, W, pe_rows=3, pe_cols=3, precision="int8"):
    precision = q.normalize_precision(precision)
    quant_mode = q.precision_mode(precision)
    K, N = W.shape

    k_per_pe = triton.cdiv(K, pe_rows)
    n_per_pe = triton.cdiv(N, pe_cols)

    BLOCK_K = triton.next_power_of_2(k_per_pe)
    BLOCK_N = triton.next_power_of_2(n_per_pe)

    partial = torch.empty(
        (pe_rows, pe_cols, BLOCK_N),
        device=x.device,
        dtype=q.precision_storage_dtype(precision),
    )

    if precision in ("fp32", "fp16"):
        partial_scales = None
        scale_buffer = torch.empty((1,), device=x.device, dtype=torch.float32)
    else:
        partial_scales = torch.empty(
            (pe_rows, pe_cols),
            device=x.device,
            dtype=torch.float32,
        )
        scale_buffer = partial_scales

    y = torch.empty((N,), device=x.device, dtype=torch.float32)

    pe_partial_gemv_kernel[(pe_rows, pe_cols)](
        x, W, partial, scale_buffer,
        K, N,
        pe_cols,
        k_per_pe,
        n_per_pe,
        BLOCK_K,
        BLOCK_N,
        quant_mode,
    )

    pe_reduce_kernel[(pe_cols,)](
        partial, scale_buffer, y,
        N,
        pe_rows,
        pe_cols,
        n_per_pe,
        BLOCK_N,
        quant_mode,
    )

    return y, partial, partial_scales

def main(pe_rows=3, pe_cols=3, precision="int8", matrix_size=(64, 64), verbose=True):

    if verbose:
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))

    torch.manual_seed(0)

    K = matrix_size[0]
    N = matrix_size[1]

    x = torch.randn((K,), device="cuda", dtype=torch.float32)
    W = torch.randn((K, N), device="cuda", dtype=torch.float32)

    y_triton, partial, partial_scales = triton_pe_gemv(x, W, pe_rows, pe_cols, precision)
    y_expected, _, expected_scales = reference.quantized_pe_gemv(
        x, W, pe_rows, pe_cols, precision
    )
    y_fp32 = x.float() @ W.float()

    triton_error = (y_triton - y_expected).abs().max().item()
    quant_error = (y_expected - y_fp32).abs().max().item()
    scale_error = None
    if partial_scales is not None:
        scale_error = (partial_scales - expected_scales).abs().max().item()

    if verbose:
        print("partial shape:", partial.shape)
        print("partial dtype:", partial.dtype)
        print("partial scales shape:", None if partial_scales is None else partial_scales.shape)
        print("max error vs quantized expected:", triton_error)
        print("max error vs fp32:", quant_error)
        print("max scale error:", scale_error)

    if precision == "fp32":
        torch.testing.assert_close(y_triton, y_expected, rtol=0.0, atol=0.0)
    elif precision == "fp16":
        torch.testing.assert_close(y_triton, y_expected, rtol=1e-3, atol=1e-2)
    else:
        torch.testing.assert_close(y_triton, y_expected, rtol=1e-5, atol=5e-5)

    if verbose:
        print("PASS")
        print(y_triton[:8])
        print(y_expected[:8])

    return triton_error, quant_error, scale_error


if __name__ == "__main__":
    main()
