import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import triton
import triton.language as tl
from . import quantization as q
from . import torch_pe_gemm as reference

@triton.jit
def pe_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PE_DIM: tl.constexpr,
    M_PER_PE: tl.constexpr,
    N_PER_PE: tl.constexpr,
    K_PER_PE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_MODE: tl.constexpr,
):
    pe_row = tl.program_id(0)
    pe_col = tl.program_id(1)

    m_local = tl.arange(0, BLOCK_M)
    n_local = tl.arange(0, BLOCK_N)
    k_local = tl.arange(0, BLOCK_K)

    m_offsets = pe_row * M_PER_PE + m_local
    n_offsets = pe_col * N_PER_PE + n_local
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for step in range(PE_DIM):
        k_offsets = step * K_PER_PE + k_local

        x_mask = (m_local[:, None] < M_PER_PE) & (m_offsets[:, None] < M) & (k_local[None, :] < K_PER_PE) & (k_offsets[None, :] < K)
        w_mask = (k_local[:, None] < K_PER_PE) & (k_offsets[:, None] < K) & (n_local[None, :] < N_PER_PE) & (n_offsets[None, :] < N)

        x_tile_vals = tl.load(X_ptr + m_offsets[:, None] * K + k_offsets[None, :],
            mask=x_mask,
            other=0.0
        ).to(tl.float32)

        w_tile_vals = tl.load(W_ptr + k_offsets[:, None] * N + n_offsets[None, :],
            mask=w_mask,
            other=0.0
        ).to(tl.float32)

        if QUANT_MODE == 0:
            q_x_tile_vals = q.quantize_fp16_tl(x_tile_vals)
            x_tile_vals = q.dequantize_fp16_tl(q_x_tile_vals)

            q_w_tile_vals = q.quantize_fp16_tl(w_tile_vals)
            w_tile_vals = q.dequantize_fp16_tl(q_w_tile_vals)
        elif QUANT_MODE == 1:
            x_tile_scale = q.symmetric_scale_2d_tl(x_tile_vals, x_mask, 4, 1.0e-8)
            q_x_tile_vals = q.quantize_symmetric_tl(x_tile_vals, x_tile_scale, 4)
            x_tile_vals = q.dequantize_symmetric_tl(q_x_tile_vals, x_tile_scale)

            w_tile_scale = q.symmetric_scale_2d_tl(w_tile_vals, w_mask, 4, 1.0e-8)
            q_w_tile_vals = q.quantize_symmetric_tl(w_tile_vals, w_tile_scale, 4)
            w_tile_vals = q.dequantize_symmetric_tl(q_w_tile_vals, w_tile_scale)
        else:
            x_tile_scale = q.symmetric_scale_2d_tl(x_tile_vals, x_mask, 8, 1.0e-8)
            q_x_tile_vals = q.quantize_symmetric_tl(x_tile_vals, x_tile_scale, 8)
            x_tile_vals = q.dequantize_symmetric_tl(q_x_tile_vals, x_tile_scale)

            w_tile_scale = q.symmetric_scale_2d_tl(w_tile_vals, w_mask, 8, 1.0e-8)
            q_w_tile_vals = q.quantize_symmetric_tl(w_tile_vals, w_tile_scale, 8)
            w_tile_vals = q.dequantize_symmetric_tl(q_w_tile_vals, w_tile_scale)

        accumulator += tl.dot(x_tile_vals, w_tile_vals, input_precision="ieee")

    y_mask = (m_local[:, None] < M_PER_PE) & (m_offsets[:, None] < M) & (n_local[None, :] < N_PER_PE) & (n_offsets[None, :] < N)
    tl.store(
        Y_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        accumulator,
        mask=y_mask,
    )




def triton_pe_gemm(X, W, pe_dim=3, precision="int8"):
    precision = q.normalize_precision(precision)
    quant_mode = q.precision_mode(precision)
    M, K = X.shape
    _, N = W.shape

    m_per_pe = triton.cdiv(M, pe_dim)
    k_per_pe = triton.cdiv(K, pe_dim)
    n_per_pe = triton.cdiv(N, pe_dim)

    BLOCK_M = triton.next_power_of_2(m_per_pe)
    BLOCK_N = triton.next_power_of_2(n_per_pe)
    BLOCK_K = triton.next_power_of_2(k_per_pe)

    Y = torch.empty((M, N), device=X.device, dtype=torch.float32)

    pe_gemm_kernel[(pe_dim, pe_dim)](
        X, W, Y,
        M, N, K,
        pe_dim,
        m_per_pe,
        n_per_pe,
        k_per_pe,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        quant_mode,
    )

    return Y


def main(pe_dim=3, precision="int8", X_matrix_size=(64, 128), W_matrix_size=(128, 64)):
    M, K = X_matrix_size
    W_K, N = W_matrix_size

    if K != W_K:
        raise ValueError(
            f"Incompatible GEMM shapes: X is {X_matrix_size} and W is {W_matrix_size}"
        )

    torch.manual_seed(0)

    X = torch.randn((M, K), device="cuda", dtype=torch.float32)
    W = torch.randn((K, N), device="cuda", dtype=torch.float32)

    C_triton = triton_pe_gemm(X, W, pe_dim=pe_dim, precision=precision)
    C = reference.quantized_pe_gemm(X, W, pe_dim=pe_dim, precision=precision)
    C_fp32 = X @ W

    print("C shape:", C.shape)
    print("max error vs reference:", (C_triton - C).abs().max().item())
    print("max error vs fp32:", (C - C_fp32).abs().max().item())
    return C_triton


if __name__ == "__main__":
    main()
