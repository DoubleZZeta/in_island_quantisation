import torch
import triton
import triton.language as tl


SUPPORTED_PRECISIONS = ("fp16", "int8", "int4")


def normalize_precision(precision):
    if isinstance(precision, int):
        precision = f"int{precision}" if precision in (4, 8) else f"fp{precision}"

    precision = precision.lower()
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision {precision!r}; expected one of {SUPPORTED_PRECISIONS}"
        )
    return precision


def precision_num_bits(precision):
    precision = normalize_precision(precision)
    return 4 if precision == "int4" else 8 if precision == "int8" else None


def precision_mode(precision):
    precision = normalize_precision(precision)
    return {"fp16": 0, "int4": 1, "int8": 2}[precision]


def precision_storage_dtype(precision):
    precision = normalize_precision(precision)
    return torch.float16 if precision == "fp16" else torch.int8


def quantize(x, precision="int8"):
    precision = normalize_precision(precision)
    if precision == "fp16":
        return quantize_fp16(x), None
    return quantize_symmetric(x, num_bits=precision_num_bits(precision))


def dequantize(x, scale, precision="int8"):
    precision = normalize_precision(precision)
    if precision == "fp16":
        return dequantize_fp16(x)
    return dequantize_symmetric(x, scale)


def signed_quant_bounds(num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    return qmin, qmax


def get_symmetric_scale(x, num_bits=8, eps=1.0e-8):
    _, qmax = signed_quant_bounds(num_bits)
    return torch.clamp(x.abs().max() / qmax, min=eps)


def round_to_nearest(x):
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def quantize_symmetric(x, num_bits=8, eps=1.0e-8):
    qmin, qmax = signed_quant_bounds(num_bits)
    scale = get_symmetric_scale(x, num_bits, eps)
    q = round_to_nearest(x / scale)
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.int8), scale


def dequantize_symmetric(q, scale):
    return q.float() * scale


def quantize_fp16(x):
    return x.to(torch.float16)


def dequantize_fp16(x):
    return x.to(torch.float32)


def quantize_dequantize_fp16(x):
    q = quantize_fp16(x)
    return dequantize_fp16(q), q

@triton.jit
def round_to_nearest_tl(x):
    return tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def symmetric_scale_tl(x, valid_mask, num_bits: tl.constexpr, eps: tl.constexpr):
    qmax = (2 ** (num_bits - 1)) - 1
    max_abs = tl.max(tl.where(valid_mask, tl.abs(x), 0.0), axis=0)
    return tl.maximum(max_abs / qmax, eps)


@triton.jit
def symmetric_scale_2d_tl(x, valid_mask, num_bits: tl.constexpr, eps: tl.constexpr):
    qmax = (2 ** (num_bits - 1)) - 1
    masked_abs = tl.where(valid_mask, tl.abs(x), 0.0)
    row_max = tl.max(masked_abs, axis=1)
    max_abs = tl.max(row_max, axis=0)
    return tl.maximum(max_abs / qmax, eps)


@triton.jit
def quantize_symmetric_tl(x, scale, num_bits: tl.constexpr):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    return tl.clamp(round_to_nearest_tl(x / scale), qmin, qmax).to(tl.int8)


@triton.jit
def dequantize_symmetric_tl(q, scale):
    return q.to(tl.float32) * scale


@triton.jit
def quantize_fp16_tl(x):
    return x.to(tl.float16)


@triton.jit
def dequantize_fp16_tl(x):
    return x.to(tl.float32)
