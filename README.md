# Project Description
This project builds a GPU-based simulator for studying low-bit LLM inference on Cerebras-like spatial accelerators.

The goal is to emulate the numerical behavior of Cerebras-style execution using PyTorch and Triton, so that different quantization schemes can be tested without needing to run full token generation directly on Cerebras hardware.

In the current simulator, computation is mapped onto a grid of processing elements (PEs). Each PE computes a local result, and the activation-like values transferred between PEs during collective communication are quantized and later dequantized during reduction. This models communication-time in-island quantization rather than only quantizing the original input or weight tensors.

# Progress/Log

## Week 1
Explored Cerebras hardware concepts and the csl programing language. 

Went rough all the tutorials and implemented a simply quantization pipeline between 2 PEs

## Week 2
Just realized there is a direction shift. Installed the Triton GPU programming language due to the fact TileLang being unstable

Implemented a Triton-based simulator for doing in-island quantizing GEMV operations among m x n PEs

Implemented a test script for inspecting the accuracy of the triton GEMV quantization process

Implemented a Triton-based simulator for doing in-island quantizing GEMM operations among n x n PEs

## Week3
Implemented a test script for inspecting the accuracy of the triton GEMM quantization process


