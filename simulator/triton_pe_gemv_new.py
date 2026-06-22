import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import triton
import triton.language as tl
from . import quantization as q
from . import torch_pe_gemv as reference

@triton.jit
def pe_partial_gemv_kernel(
    x_ptr, W_ptr, b_ptr, partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    PE_COLS: tl.constexpr,
    M_PER_PE: tl.constexpr,
    N_PER_PE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pe_row = tl.program_id(0)
    pe_col = tl.program_id(1)

    m_local = tl.arange(0, BLOCK_M)
    n_local = tl.arange(0, BLOCK_N)

    m_offsets = pe_row * M_PER_PE + m_local
    n_offsets = pe_col * N_PER_PE + n_local

    x_vals = tl.load(
        x_ptr + n_offsets,
        mask=(n_local < N_PER_PE) & (n_offsets < N),
        other=0.0,
    ).to(tl.float32)

    W_vals = tl.load(
        W_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        mask=(
            (m_local[:, None] < M_PER_PE)
            & (m_offsets[:, None] < M)
            & (n_local[None, :] < N_PER_PE)
            & (n_offsets[None, :] < N)
        ),
        other=0.0,
    ).to(tl.float32)

    # acc = tl.sum(W_vals * x_vals[None, :], axis=1)
    valid_m = (m_local < M_PER_PE) & (m_offsets < M)
    acc = tl.load(b_ptr + m_offsets, mask=valid_m, other=0.0)
    if pe_col != 0:
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, BLOCK_N):
        n_offset = pe_col * N_PER_PE + n
        valid_n = (n < N_PER_PE) & (n_offset < N)

        x_val = tl.load(
            x_ptr + n_offset,
            mask=valid_n,
            other=0.0,
        )

        w_vals = tl.load(
            W_ptr + m_offsets * N + n_offset,
            mask=valid_m & valid_n,
            other=0.0,
        )

        acc += w_vals * x_val

    partial_base = (pe_row * PE_COLS + pe_col) * BLOCK_M

    tl.store(
        partial_ptr + partial_base + m_local,
        acc,
        mask=valid_m,
    )


@triton.jit
def pe_reduce_kernel(
    partial_ptr, y_ptr,
    M: tl.constexpr,
    PE_ROWS: tl.constexpr,
    PE_COLS: tl.constexpr,
    M_PER_PE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pe_row = tl.program_id(0)

    m_local = tl.arange(0, BLOCK_M)
    m_offsets = pe_row * M_PER_PE + m_local

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for pe_col in range(0, PE_COLS):
        partial_base = (pe_row * PE_COLS + pe_col) * BLOCK_M
        vals = tl.load(
            partial_ptr + partial_base + m_local,
            mask=(m_local < M_PER_PE) & (m_offsets < M),
            other=0,
        )
        acc += vals

    tl.store(
        y_ptr + m_offsets,
        acc,
        mask=(m_local < M_PER_PE) & (m_offsets < M),
    )


def triton_pe_gemv(x, W, b, pe_rows=3, pe_cols=3):
    # implements Ax
    # x: (N,)
    # W: (M, N)

    M, N = W.shape

    m_per_pe = triton.cdiv(M, pe_rows)
    n_per_pe = triton.cdiv(N, pe_cols)

    BLOCK_M = triton.next_power_of_2(m_per_pe)
    BLOCK_N = triton.next_power_of_2(n_per_pe)

    partial = torch.empty(
        (pe_rows, pe_cols, BLOCK_M),
        device=x.device,
        dtype=torch.float32,)

    y = torch.empty((M,), device=x.device, dtype=torch.float32)

    pe_partial_gemv_kernel[(pe_rows, pe_cols)](
        x, W, b, partial,
        M, N,
        pe_cols,
        m_per_pe,
        n_per_pe,
        BLOCK_M,
        BLOCK_N,
    )

    pe_reduce_kernel[(pe_rows,)](
        partial, y,
        M,
        pe_rows,
        pe_cols,
        m_per_pe,
        BLOCK_M,
    )

    return y, partial
