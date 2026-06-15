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

    M, K = X.shape
    _, N = W.shape
    m_per_pe = (M + pe_dim - 1) // pe_dim
    k_per_pe = (K + pe_dim - 1) // pe_dim
    n_per_pe = (N + pe_dim - 1) // pe_dim

    x_tiles = []
    for pe_row in range(pe_dim):
        m_start = pe_row * m_per_pe
        m_end = min(m_start + m_per_pe, M)
        tile_row = []
        for k_step in range(pe_dim):
            k_start = k_step * k_per_pe
            k_end = min(k_start + k_per_pe, K)
            tile_row.append(X[m_start:m_end, k_start:k_end])
        x_tiles.append(tile_row)

    w_tiles = []
    for k_step in range(pe_dim):
        k_start = k_step * k_per_pe
        k_end = min(k_start + k_per_pe, K)
        tile_row = []
        for pe_col in range(pe_dim):
            n_start = pe_col * n_per_pe
            n_end = min(n_start + n_per_pe, N)
            tile_row.append(W[k_start:k_end, n_start:n_end])
        w_tiles.append(tile_row)

    m_sizes = [min((pe_row + 1) * m_per_pe, M) - pe_row * m_per_pe for pe_row in range(pe_dim)]
    n_sizes = [min((pe_col + 1) * n_per_pe, N) - pe_col * n_per_pe for pe_col in range(pe_dim)]

    Y_tiles = [
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

                Y_tiles[pe_row][pe_col] += x_available @ w_available

    Y = torch.cat(
        [torch.cat(tile_row, dim=1) for tile_row in Y_tiles],
        dim=0,
    )

    return Y

    
