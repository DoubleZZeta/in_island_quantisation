# A torch based simulator for the PE GEMV operation, used for testing and debugging the Triton implementation.
import torch
import triton
import triton.language as tl
from . import quantization as q

def quantized_pe_gemv(x, W, pe_rows=3, pe_cols=3, precision="int8"):
    precision = q.normalize_precision(precision)
    K, N = W.shape
    k_per_pe = triton.cdiv(K, pe_rows)
    n_per_pe = triton.cdiv(N, pe_cols)

    y = torch.zeros((N,), device=x.device, dtype=torch.float32)
    partials = []
    scales = []

    for pe_row in range(pe_rows):
        row_start = pe_row * k_per_pe
        row_end = min(row_start + k_per_pe, K)
        partial_row = []
        scale_row = []

        for pe_col in range(pe_cols):
            col_start = pe_col * n_per_pe
            col_end = min(col_start + n_per_pe, N)

            partial_fp = x[row_start:row_end].float() @ W[row_start:row_end, col_start:col_end].float()
            partial_q, scale = q.quantize(partial_fp, precision)
            y[col_start:col_end] += q.dequantize(partial_q, scale, precision)

            partial_row.append(partial_q)
            if scale is not None:
                scale_row.append(scale)

        partials.append(partial_row)
        if scale_row:
            scales.append(torch.stack(scale_row))

    stacked_scales = torch.stack(scales) if scales else None
    return y, partials, stacked_scales
