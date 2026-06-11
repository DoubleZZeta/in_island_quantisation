import torch
import triton
import triton.language as tl


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


def quantize_dequantize_symmetric(x, num_bits=8, eps=1.0e-8):
    q, scale = quantize_symmetric(x, num_bits, eps)
    return dequantize_symmetric(q, scale), q, scale


def get_symmetric_int8_scale(x):
    return get_symmetric_scale(x, num_bits=8)


def quantize_symmetric_int8(x):
    return quantize_symmetric(x, num_bits=8)


def dequantize_symmetric_int8(q, scale):
    return dequantize_symmetric(q, scale)


def get_symmetric_int4_scale(x):
    return get_symmetric_scale(x, num_bits=4)


def quantize_symmetric_int4(x):
    return quantize_symmetric(x, num_bits=4)


def dequantize_symmetric_int4(q, scale):
    return dequantize_symmetric(q, scale)


@triton.jit
def round_to_nearest_tl(x):
    return tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def symmetric_scale_tl(x, valid_mask, num_bits: tl.constexpr, eps: tl.constexpr):
    qmax = (2 ** (num_bits - 1)) - 1
    max_abs = tl.max(tl.where(valid_mask, tl.abs(x), 0.0), axis=0)
    return tl.maximum(max_abs / qmax, eps)


@triton.jit
def quantize_symmetric_tl(x, scale, num_bits: tl.constexpr):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    return tl.clamp(round_to_nearest_tl(x / scale), qmin, qmax).to(tl.int8)


@triton.jit
def dequantize_symmetric_tl(q, scale):
    return q.to(tl.float32) * scale
