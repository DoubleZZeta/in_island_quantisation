import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import triton
import triton.language as tl

@triton.jit
def pe_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PE_DIM: tl.constexpr,
    M_PER_PE: tl.constexpr,
    N_PER_PE: tl.constexpr,
    K_PER_PE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pe_row = tl.program_id(0)
    pe_col = tl.program_id(1)
    local_j = tl.program_id(2)

    m_local = tl.arange(0, BLOCK_M)
    m_offsets = pe_row * M_PER_PE + m_local

    n_offset = pe_col * N_PER_PE + local_j

    valid_m = (m_local < M_PER_PE) & (m_offsets < M)
    valid_j = (local_j < N_PER_PE) & (n_offset < N)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for step in range(PE_DIM):
        for k in range(0, BLOCK_K):
            k_offset = step * K_PER_PE + k
            valid_k = (k < K_PER_PE) & (k_offset < K)

            a_vals = tl.load(
                A_ptr + m_offsets * K + k_offset,
                mask=valid_m & valid_k,
                other=0.0,
            ).to(tl.float32)

            b_val = tl.load(
                B_ptr + k_offset * N + n_offset,
                mask=valid_k & valid_j,
                other=0.0,
            ).to(tl.float32)

            acc += a_vals * b_val

    tl.store(
        C_ptr + m_offsets * N + n_offset,
        acc,
        mask=valid_m & valid_j,
    )

def triton_pe_gemm(A, B, pe_dim=3):
    M, K = A.shape
    _, N = B.shape

    m_per_pe = triton.cdiv(M, pe_dim)
    k_per_pe = triton.cdiv(K, pe_dim)
    n_per_pe = triton.cdiv(N, pe_dim)

    BLOCK_M = triton.next_power_of_2(m_per_pe)
    BLOCK_K = max(16, triton.next_power_of_2(k_per_pe))

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    pe_gemm_kernel[(pe_dim, pe_dim, n_per_pe)](
        A, B, C,
        M, N, K,
        pe_dim,
        m_per_pe,
        n_per_pe,
        k_per_pe,
        BLOCK_M,
        BLOCK_K,
    )

    return C