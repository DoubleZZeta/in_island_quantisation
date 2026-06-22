import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import csv
import sys
import torch
import numpy as np
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import simulator.triton_pe_gemv_new as triton_pe_gemv


CASE_DIR = Path(__file__).resolve().parent / "gemv_case"
CASE_DIR.mkdir(exist_ok=True)
GEMV_DIR = PROJECT_ROOT / "simulator" / "gemv"
COMMAND = GEMV_DIR / "commands_wse2.sh"
RESULT_PATH = Path(__file__).resolve().parent / "triton_csl_gemv_results.csv"
CEREBRAS_TIMEOUT_SECONDS = 120

kernel_x_dims = [i for i in range(1,11)]
kernel_y_dims = [i for i in range(1,11)]
# Ms = [i for i in range(16, 65, 16)]
# Ns = [i for i in range(16, 65, 16)]
Ms = [24]
Ns = [32]

def can_run(M, N, kernel_x_dim, kernel_y_dim):
    if kernel_x_dim < 2:
        return False
    if M % kernel_y_dim != 0 or N % kernel_x_dim != 0:
        return False
    return True


def run_cerebras(case_dir, M, N, kernel_x_dim, kernel_y_dim):
    inputs_path = case_dir / "inputs.npz"
    y_cerebras_path = case_dir / "y_cerebras.npy"
    #out_dir = f"out_M{M}_N{N}_kx{kernel_x_dim}_ky{kernel_y_dim}"
    out_dir = "out"

    subprocess.run(
        [
            str(COMMAND),
            "--kernel-x-dim", str(kernel_x_dim),
            "--kernel-y-dim", str(kernel_y_dim),
            "--M", str(M),
            "--N", str(N),
            "--out-dir", out_dir,
            "--inputs", str(inputs_path),
            "--y-out", str(y_cerebras_path),
        ],
        cwd=GEMV_DIR,
        check=True,
        timeout=CEREBRAS_TIMEOUT_SECONDS,
    )

    return np.load(y_cerebras_path)


def run_triton(A, x, b, kernel_x_dim, kernel_y_dim):
    x_torch = torch.from_numpy(x).cuda()
    W_torch = torch.from_numpy(A.copy()).cuda()
    b_torch = torch.from_numpy(b).cuda()

    y_torch, _ = triton_pe_gemv.triton_pe_gemv(
        x_torch,
        W_torch,
        b_torch,
        pe_rows=kernel_y_dim,
        pe_cols=kernel_x_dim,
    )

    return y_torch.detach().cpu().numpy()


def run_tests():
    rng = np.random.default_rng(0)
    failures = []

    with RESULT_PATH.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "M",
            "N",
            "Kernel X Dim",
            "Kernel Y Dim",
            "Max Abs Error",
            "Exact Equal",
            "Status",
        ])

        for M in Ms:
            for N in Ns:
                for kernel_x_dim in kernel_x_dims:
                    for kernel_y_dim in kernel_y_dims:
                        if not can_run(M, N, kernel_x_dim, kernel_y_dim):
                            continue

                        case_name = f"M{M}_N{N}_kx{kernel_x_dim}_ky{kernel_y_dim}"
                        case_dir = CASE_DIR / case_name
                        case_dir.mkdir(parents=True, exist_ok=True)

                        A = rng.random((M, N), dtype=np.float32)
                        x = rng.random((N,), dtype=np.float32)
                        b = rng.random((M,), dtype=np.float32)
                        np.savez(case_dir / "inputs.npz", A=A, x=x, b=b)

                        print(f"Running {case_name}")
                        try:
                            y_cerebras = run_cerebras(
                                case_dir,
                                M,
                                N,
                                kernel_x_dim,
                                kernel_y_dim,
                            )
                            y_triton = run_triton(
                                A,
                                x,
                                b,
                                kernel_x_dim,
                                kernel_y_dim,
                            )
                            np.save(case_dir / "y_triton.npy", y_triton)

                            max_abs_error = np.max(np.abs(y_cerebras - y_triton)).item()
                            exact_equal = np.array_equal(y_cerebras, y_triton)
                            status = "PASS" if exact_equal else "DIFF"
                        except Exception as error:
                            max_abs_error = ""
                            exact_equal = ""
                            status = f"FAIL: {type(error).__name__}: {' '.join(str(error).split())}"
                            failures.append((case_name, status))
                            print(status)

                        writer.writerow([
                            M,
                            N,
                            kernel_x_dim,
                            kernel_y_dim,
                            max_abs_error,
                            exact_equal,
                            status,
                        ])

    if failures:
        raise SystemExit(f"{len(failures)} test case(s) failed; see {RESULT_PATH}")



if __name__ == "__main__":
    run_tests()
