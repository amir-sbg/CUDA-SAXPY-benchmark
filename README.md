# CUDA SAXPY Benchmark

A CUDA C++ benchmark for a memory-bound vector operation used throughout numerical computing and ML systems:

```text
y = alpha * x + y
```

The same update runs on the CPU and in a custom CUDA kernel. The executable checks numerical agreement, measures kernel and transfer time, and reports bandwidth, throughput, launch geometry, and error statistics. An `sgd-step` workload reuses the kernel for the equivalent optimizer update:

```text
weights = weights - learning_rate * gradients
```

## What it measures

- CPU time, host-to-device time, kernel time, device-to-host time, and end-to-end time
- kernel and end-to-end nanoseconds per vector element
- effective device and transfer bandwidth, GFLOP/s, and arithmetic intensity
- block count, working-set size, and launched threads per SM
- maximum and mean absolute CPU/GPU error
- block-size and vector-size sweep summaries

The kernel uses coalesced one-dimensional accesses and a grid-stride loop, so the same implementation handles vectors larger than the active grid.

## Build

Requirements: an NVIDIA GPU, CUDA Toolkit with `nvcc`, CMake 3.24+, and a C++17 compiler.

```bash
git clone https://github.com/amir-sbg/CUDA-SAXPY-benchmark.git
cd CUDA-SAXPY-benchmark
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Run

```bash
./build/cuda_saxpy \
  --elements 16777216 \
  --iterations 100 \
  --warmup 1 \
  --block-size 256 \
  --alpha 2.0 \
  --workload saxpy \
  --tolerance 1e-5 \
  --json-output reports/saxpy.json \
  --csv-output reports/saxpy.csv
```

For the optimizer-style update:

```bash
./build/cuda_saxpy --workload sgd-step --learning-rate 0.001
```

The process returns a nonzero status when the GPU result exceeds the configured tolerance. JSON and CSV outputs are useful for comparing launch configurations without mixing timing and correctness checks.

## Sweeps

```bash
python3 scripts/run_block_sweep.py \
  --elements 1048576 4194304 16777216 \
  --blocks 128 256 512 \
  --repeats 5 \
  --workload sgd-step \
  --output reports/block_sweep.csv

python3 scripts/summarize_sweep.py \
  --input reports/block_sweep.csv \
  --output reports/block_sweep.md \
  --peak-bandwidth-gbps 1008
```

Repeated runs are aggregated by configuration. The summary selects the block size with the highest median bandwidth and reports timing IQR and bandwidth coefficient of variation, which makes noisy launch configurations easier to spot. Supplying the GPU's theoretical memory bandwidth also reports the fraction reached by this memory-bound kernel.

## Project layout

```text
src/saxpy.cu                 CUDA kernel, host reference, timing, and CLI
scripts/run_block_sweep.py   block/vector-size sweep runner
scripts/summarize_sweep.py   CSV-to-Markdown summary
CMakeLists.txt               CUDA build configuration
Makefile                     build and run shortcuts
```
