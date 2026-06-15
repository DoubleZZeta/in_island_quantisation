import torch
from . import quantization as q


def communicate(tile, precision):
    tile_q, scale = q.quantize(tile, precision)
    return q.dequantize(tile_q, scale, precision)


def quantized_pe_gemm(X, W, pe_dim=3, precision="int8"):
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X and W must both be 2D matrices")
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible GEMM shapes: X is {tuple(X.shape)} and W is {tuple(W.shape)}"
        )

    x_row_tiles = torch.tensor_split(X, pe_dim, dim=0)
    x_tiles = [torch.tensor_split(x_row, pe_dim, dim=1) for x_row in x_row_tiles]

    w_k_tiles = torch.tensor_split(W, pe_dim, dim=0)
    w_tiles = [
        torch.tensor_split(w_k, pe_dim, dim=1)
        for w_k in w_k_tiles
    ]

    m_sizes = [x_row.shape[0] for x_row in x_row_tiles]

    w_col_tiles = torch.tensor_split(W, pe_dim, dim=1)
    n_sizes = [w_col.shape[1] for w_col in w_col_tiles]

    C_tiles = [
        [
            torch.zeros(
                (m_sizes[pe_row], n_sizes[pe_col]),
                device=X.device,
                dtype=torch.float32,
            )
            for pe_col in range(pe_dim)
        ]
        for pe_row in range(pe_dim)
    ]

    for k_step in range(pe_dim):
        for pe_row in range(pe_dim):
            for pe_col in range(pe_dim):
                x_source = x_tiles[pe_row][k_step]
                w_source = w_tiles[k_step][pe_col]

                x_available = communicate(x_source, precision)
                w_available = communicate(w_source, precision)

                C_tiles[pe_row][pe_col] += x_available @ w_available
    
    C = torch.cat(
        [torch.cat(tile_row, dim=1) for tile_row in C_tiles],
        dim=0,
    )

    return C

    
