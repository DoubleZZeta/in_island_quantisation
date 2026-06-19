import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import simulator.triton_pe_gemv as triton_pe_gemv

pe_grids = [i for i in range(1,5)]
matrix_sizes = [(64, 64), (63, 65), (128, 128), (512, 512), (512, 2048)]
precisions = ["fp16", "int4", "int8"]

def run_tests():
    result_path = Path(__file__).with_name("pe_gemv_results.csv")
    failures = []

    with result_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "PE Grid",
            "X Matrix Size",
            "M Matrix Size",
            "Precision",
            "Triton vs Reference Error",
            "Quantization vs FP32 Error",
            "Scale Error",
            "Status",
        ])

        for pe_grid in pe_grids:
            for matrix_size in matrix_sizes:
                for precision in precisions:
                    case = (pe_grid, matrix_size, precision)
                    print(
                        f"Running PE Grid: {pe_grid}x{pe_grid}, "
                        f"Matrix Size: {matrix_size}, Precision: {precision}"
                    )
                    try:
                        triton_error, quant_error, scale_error = triton_pe_gemv.main(
                            pe_rows=pe_grid,
                            pe_cols=pe_grid,
                            precision=precision,
                            matrix_size=matrix_size,
                            verbose=False,
                        )
                        status = "PASS"
                    except Exception as error:
                        triton_error = ""
                        quant_error = ""
                        scale_error = ""
                        error_message = " ".join(str(error).split())
                        status = f"FAIL: {type(error).__name__}: {error_message}"   
                        failures.append((case, status))
                        print(status)

                    writer.writerow([
                        f"{pe_grid}x{pe_grid}",
                        f"{matrix_size[0]}x{matrix_size[1]}",
                        precision,
                        triton_error,
                        quant_error,
                        scale_error,
                        status,
                    ])

    if failures:
        raise SystemExit(f"{len(failures)} test case(s) failed; see {result_path}")

    print(f"All tests passed; results written to {result_path}")


if __name__ == "__main__":
    run_tests()
