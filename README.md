# CUDA SAXPY Benchmark

A small CUDA C++ project that implements and benchmarks the SAXPY operation:

```text
output[i] = alpha * x[i] + y[i]
```

The same operation is implemented on the CPU and in a custom CUDA kernel. The program checks the results, measures the CPU implementation with `std::chrono`, and measures GPU kernel time with CUDA events.

## What it covers

- host-to-device and device-to-host memory transfers
- device memory ownership with a small RAII wrapper
- a `__global__` kernel using grid-stride indexing
- configurable block size and vector length
- CUDA runtime error checking
- correctness comparison against the CPU reference
- repeated kernel timing and a simple speedup estimate
- effective device-memory bandwidth derived from the timed kernel
- separate host-to-device, kernel, device-to-host, and end-to-end timing
- achieved GFLOP/s, arithmetic intensity, and launched block-count reporting

The kernel uses coalesced one-dimensional accesses. A grid-stride loop allows the same kernel to handle vectors larger than the number of resident threads while the report records how many blocks were launched for the selected vector size and GPU.

## Requirements

- NVIDIA GPU with a supported compute capability
- CUDA Toolkit with `nvcc`
- CMake 3.24 or newer
- C++17 compiler

## Build

```bash
git clone https://github.com/amir-sbg/CUDA.git
cd CUDA

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The Makefile provides the same commands:

```bash
make build
```

## Run

```bash
./build/cuda_saxpy
```

Available options:

```bash
./build/cuda_saxpy \
  --elements 16777216 \
  --iterations 100 \
  --warmup 1 \
  --block-size 256 \
  --alpha 2.0 \
  --seed 7 \
  --json-output reports/saxpy.json
```

The program prints the selected GPU, vector size, block size, launched blocks, average CPU time, H2D copy time, average GPU kernel time, D2H copy time, end-to-end GPU time, kernel speedup, end-to-end speedup, effective bandwidth, achieved GFLOP/s, arithmetic intensity, and maximum absolute error. `--warmup` controls the number of untimed kernel launches before CUDA-event timing; it defaults to one. `--json-output` writes the same benchmark summary to a JSON file and creates its parent directory when needed. The program returns a nonzero status when the result differs from the CPU reference by more than `1e-5`.

## Timing note

The kernel-time metrics isolate device execution. The transfer and end-to-end fields show the cost of moving inputs and outputs across PCIe/NVLink, which is often the dominant cost for a bandwidth-bound vector operation like SAXPY.

## Project structure

```text
.
├── src/saxpy.cu       # host code, CUDA kernel, timing, and CLI
├── CMakeLists.txt     # CUDA build configuration
├── Makefile           # build and run shortcuts
└── README.md
```
