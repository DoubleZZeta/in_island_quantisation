# triton_gemm.py

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            A + offs_m[:, None] * K + (k0 + offs_k[None, :]),
            mask=(offs_m[:, None] < M) & (k0 + offs_k[None, :] < K),
            other=0.0,
        )

        b = tl.load(
            B + (k0 + offs_k[:, None]) * N + offs_n[None, :],
            mask=(k0 + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b)

    tl.store(
        C + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul(A, B):
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16
    assert B.dtype == torch.float16
    assert A.shape[1] == B.shape[0]

    M, K = A.shape
    K, N = B.shape

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return C


def main():
    torch.manual_seed(0)

    M = 64
    K = 64
    N = 64

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)

    C = matmul(A, B)
    C_ref = A @ B

    print("max error:", (C - C_ref).abs().max().item())

    torch.testing.assert_close(
        C,
        C_ref.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    print("PASS")
    print(C[:4, :4])


if __name__ == "__main__":
    main()