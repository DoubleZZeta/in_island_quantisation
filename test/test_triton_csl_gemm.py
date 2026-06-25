import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import csv
import sys
import torch
import numpy as np
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import simulator.triton_pe_gemm_new as triton_pe_gemm


CASE_DIR = Path(__file__).resolve().parent / "gemm_case"
CASE_DIR.mkdir(exist_ok=True)
GEMM_DIR = PROJECT_ROOT / "simulator" / "gemm"
COMMAND = GEMM_DIR / "commands_wse2.sh"
RESULT_PATH = Path(__file__).resolve().parent / "triton_csl_gemm_results.csv"
CEREBRAS_TIMEOUT_SECONDS = 120

kernel_dims = [i for i in range(2,11)] # for x and y
Ms = [i for i in range(16, 65, 16)]
Ns = [i for i in range(16, 65, 16)]
Ks = [i for i in range(16, 65, 16)]
# kernel_dims = [2]
# Ms = [2]
# Ns = [24]
# Ks = [16]

def can_run(M, N, K, kernel_dim):
    if M % kernel_dim != 0 or N % kernel_dim != 0 or K % kernel_dim != 0:
        return False
    return True


def run_cerebras(case_dir, Mt, Nt, Kt, kernel_dim):
    inputs_path = case_dir / "inputs.npz"
    c_cerebras_path = case_dir / "c_cerebras.npy"
    #out_dir = f"out_M{M}_N{N}_kx{kernel_x_dim}_ky{kernel_y_dim}"
    out_dir = "out"

    subprocess.run(
        [
            str(COMMAND),
            "--P", str(kernel_dim),
            "--Mt", str(Mt),
            "--Nt", str(Nt),
            "--Kt", str(Kt),
            "--out-dir", out_dir,
            "--inputs", str(inputs_path),
            "--c-out", str(c_cerebras_path),
        ],
        cwd=GEMM_DIR,
        check=True,
        timeout=CEREBRAS_TIMEOUT_SECONDS,
    )

    return np.load(c_cerebras_path)


def run_triton(A, B, kernel_dim):
    A_torch = torch.from_numpy(A.copy()).cuda()
    B_torch = torch.from_numpy(B.copy()).cuda()

    y_torch = triton_pe_gemm.triton_pe_gemm(
        A_torch,
        B_torch,
        pe_dim=kernel_dim,
    )

    return y_torch.detach().cpu().numpy()


def run_tests():
    rng = np.random.default_rng(0)
    failures = []

    with RESULT_PATH.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "M",
            "K",
            "N",
            "Kernel Dim",
            "Mt",
            "Kt",
            "Nt",
            "Max Abs Error",
            "Exact Equal",
            "Status",
        ])

        for M in Ms:
            for K in Ks:
                for N in Ns:
                    for kernel_dim in kernel_dims:
                        if not can_run(M, N, K, kernel_dim):
                            continue

                        Mt = M // kernel_dim
                        Kt = K // kernel_dim
                        Nt = N // kernel_dim

                        case_name = f"M{M}_K{K}_N{N}_P{kernel_dim}"
                        case_dir = CASE_DIR

                        A = rng.random((M, K), dtype=np.float32)
                        B = rng.random((K, N), dtype=np.float32)
                        np.savez(case_dir / "inputs.npz", A=A, B=B)

                        print(f"Running {case_name}")
                        try:
                            c_cerebras = run_cerebras(
                                case_dir,
                                Mt,
                                Nt,
                                Kt,
                                kernel_dim,
                            )
                            c_triton = run_triton(
                                A,
                                B,
                                kernel_dim,
                            )
                            np.save(case_dir / "c_triton.npy", c_triton)

                            max_abs_error = np.max(np.abs(c_cerebras - c_triton)).item()
                            exact_equal = np.array_equal(c_cerebras, c_triton)
                            status = "PASS" if exact_equal else "DIFF"
                        except Exception as error:
                            max_abs_error = ""
                            exact_equal = ""
                            status = f"FAIL: {type(error).__name__}: {' '.join(str(error).split())}"
                            failures.append((case_name, status))
                            print(status)

                        writer.writerow([
                            M,
                            K,
                            N,
                            kernel_dim,
                            Mt,
                            Kt,
                            Nt,
                            max_abs_error,
                            exact_equal,
                            status,
                        ])

    if failures:
        raise SystemExit(f"{len(failures)} test case(s) failed; see {RESULT_PATH}")



if __name__ == "__main__":
    run_tests()
