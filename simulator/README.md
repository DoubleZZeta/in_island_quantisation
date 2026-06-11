# Project Description
This project aims to explore the performance of low-bit LLMs that runs on Cerebras-like hardwares via GPU based simulation

# Progress

## Week 1
Explored Cerebras hardware concepts and the csl programing language. 

Went rough all the tutorials and implemented a simply quantization pipeline between 2 PEs

## Week 2
Just realized there is a direction shift. Installed the Triton GPU programming language due to the fact TileLang being unstable

Implemented a Triton-based simulator for doing in-island quantizing GEMV operations among 3 x 3 PEs