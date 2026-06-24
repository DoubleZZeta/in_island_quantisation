import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import triton
import triton.language as tl

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

        accumulator += tl.dot(x_tile_vals, w_tile_vals, input_precision="ieee")

    y_mask = (m_local[:, None] < M_PER_PE) & (m_offsets[:, None] < M) & (n_local[None, :] < N_PER_PE) & (n_offsets[None, :] < N)
    tl.store(
        Y_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        accumulator,
        mask=y_mask,
    )




def triton_pe_gemm(X, W, pe_dim=3):
    M, K = X.shape
    _, N = W.shape

    m_per_pe = triton.cdiv(M, pe_dim)
    k_per_pe = triton.cdiv(K, pe_dim)
    n_per_pe = triton.cdiv(N, pe_dim)

    BLOCK_M = triton.next_power_of_2(m_per_pe)
    BLOCK_N = triton.next_power_of_2(n_per_pe)
    BLOCK_K = max(16, triton.next_power_of_2(k_per_pe))

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
    )

    return Y
