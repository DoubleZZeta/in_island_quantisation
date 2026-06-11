func = matmul(1024, 1024, 1024, 128, 128, 32)
print(func)  # Prints an IR-like representation of the TileLang kernel

artifact = tilelang.lower(func)

profiler = Profiler(artifact.rt_mod, artifact.params, result_idx=[2])

import torch
a = torch.randn(1024, 1024).cuda().half()
b = torch.randn(1024, 1024).cuda().half()

c = profiler(a, b)
ref_c = a @ b

# Validate results
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

# Get CUDA Kernel Source
print(artifact.kernel_source)