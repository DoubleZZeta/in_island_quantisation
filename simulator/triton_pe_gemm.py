import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from . import torch_pe_gemm as reference




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

    C = reference.quantized_pe_gemm(X, W, pe_dim=pe_dim, precision=precision)
    C_fp32 = X @ W

    print("C shape:", C.shape)
    print("max error vs fp32:", (C - C_fp32).abs().max().item())
    return C


if __name__ == "__main__":
    main()
