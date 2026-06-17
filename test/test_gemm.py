import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import simulator.triton_pe_gemm as triton_pe_gemm

MAX_BLOCK_SIZE = 64
MIN_DOT_K = 16

test_levels = [
    (
        "small",
        [1, 2, 3, 4],
        [
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (63, 65, 67),
            (32, 64, 128),
            (128, 64, 32),
        ],
    ),
    (
        "medium",
        [4, 6, 8],
        [
            (128, 128, 128),
            (127, 129, 131),
            (64, 256, 64),
            (256, 64, 256),
        ],
    ),
    (
        "large",
        [8, 16],
        [
            (512, 512, 512),
            (512, 1024, 512),
            (512, 2048, 512),
        ],
    ),
]
precisions = ["fp16", "int4", "int8"]


def cdiv(x, y):
    return (x + y - 1) // y


def next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def block_sizes(m, k, n, pe_dim):
    block_m = next_power_of_2(cdiv(m, pe_dim))
    block_k = max(MIN_DOT_K, next_power_of_2(cdiv(k, pe_dim)))
    block_n = next_power_of_2(cdiv(n, pe_dim))
    return block_m, block_k, block_n


def should_skip(m, k, n, pe_dim):
    return max(block_sizes(m, k, n, pe_dim)) > MAX_BLOCK_SIZE

def run_tests():
    result_path = Path(__file__).with_name("pe_gemm_results.csv")
    failures = []

    with result_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "Level",
            "PE Grid",
            "M",
            "K",
            "N",
            "X_Matrix Size",
            "W_Matrix Size",
            "Precision",
            "Block M",
            "Block K",
            "Block N",
            "Triton vs Reference Error",
            "Quantization vs FP32 Error",
            "Status",
        ])

        for level_name, pe_grids, gemm_shapes in test_levels:
            for pe_grid in pe_grids:
                for m, k, n in gemm_shapes:
                    block_m, block_k, block_n = block_sizes(m, k, n, pe_grid)
                    if should_skip(m, k, n, pe_grid):
                        status = (
                            f"SKIP: block size exceeds {MAX_BLOCK_SIZE} "
                            f"({block_m}x{block_k}x{block_n})"
                        )
                        print(
                            f"Skipping {level_name} PE Grid: {pe_grid}x{pe_grid}, "
                            f"X Matrix Size: {m}x{k}, W Matrix Size: {k}x{n}, {status}"
                        )
                        triton_error = ""
                        quant_error = ""
                        for precision in precisions:
                            writer.writerow([
                                level_name,
                                f"{pe_grid}x{pe_grid}",
                                m,
                                k,
                                n,
                                f"{m}x{k}",
                                f"{k}x{n}",
                                precision,
                                block_m,
                                block_k,
                                block_n,
                                triton_error,
                                quant_error,
                                status,
                            ])
                        continue

                    for precision in precisions:
                        case = (level_name, pe_grid, m, k, n, precision)
                        print(
                            f"Running {level_name} PE Grid: {pe_grid}x{pe_grid}, "
                            f"X Matrix Size: {m}x{k}, W Matrix Size: {k}x{n}, "
                            f"Blocks: {block_m}x{block_k}x{block_n}, Precision: {precision}"
                        )
                        try:
                            triton_error, quant_error = triton_pe_gemm.main(
                                pe_dim=pe_grid,
                                precision=precision,
                                X_matrix_size=(m, k),
                                W_matrix_size=(k, n),
                                verbose=False,
                            )
                            status = "PASS"
                        except Exception as error:
                            triton_error = ""
                            quant_error = ""
                            status = f"FAIL: {type(error).__name__}: {error}"
                            failures.append((case, status))
                            print(status)

                        writer.writerow([
                            level_name,
                            f"{pe_grid}x{pe_grid}",
                            m,
                            k,
                            n,
                            f"{m}x{k}",
                            f"{k}x{n}",
                            precision,
                            block_m,
                            block_k,
                            block_n,
                            triton_error,
                            quant_error,
                            status,
                        ])

    if failures:
        raise SystemExit(f"{len(failures)} test case(s) failed; see {result_path}")

    print(f"All tests passed; results written to {result_path}")


if __name__ == "__main__":
    run_tests()
