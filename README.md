# CUDA SAXPY Benchmark

A small CUDA C++ project that implements and benchmarks the SAXPY operation:

```text
output[i] = alpha * x[i] + y[i]
```

The same operation is implemented on the CPU and in a custom CUDA kernel. The program checks the results, measures the CPU implementation with `std::chrono`, and measures GPU kernel time with CUDA events.

The benchmark can also run the same memory-bound operation as a simple SGD update:

```text
weights[i] = weights[i] - learning_rate * gradients[i]
```

## What it covers

- host-to-device and device-to-host memory transfers
- device memory ownership with a small RAII wrapper
- a `__global__` kernel using grid-stride indexing
- configurable block size and vector length
- SAXPY and SGD-style update workloads
- CUDA runtime error checking
- device-aware launch validation against `cudaDeviceProp`
- correctness comparison against the CPU reference
- repeated kernel timing and a simple speedup estimate
- effective device-memory bandwidth derived from the timed kernel
- separate host-to-device, kernel, device-to-host, and end-to-end timing
- achieved GFLOP/s, arithmetic intensity, and launched block-count reporting
- host/device working-set size and launched threads per SM

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
  --workload saxpy \
  --tolerance 1e-5 \
  --seed 7 \
  --json-output reports/saxpy.json \
  --csv-output reports/saxpy_sweep.csv
```

For an optimizer-style memory update:

```bash
./build/cuda_saxpy --workload sgd-step --learning-rate 0.001
```

The program prints the selected GPU, workload, vector size, block size, launched blocks, working-set size, average CPU time, H2D copy time, average GPU kernel time, D2H copy time, end-to-end GPU time, kernel speedup, end-to-end speedup, effective bandwidth, achieved GFLOP/s, arithmetic intensity, tolerance, and maximum absolute error. `--warmup` controls the number of untimed kernel launches before CUDA-event timing; it defaults to one. `--json-output` writes the benchmark summary to a JSON file, while `--csv-output` appends a row that is convenient for block-size or vector-size sweeps. The program creates parent directories when needed and returns a nonzero status when the result differs from the CPU reference by more than the configured tolerance.

Example sweep:

```bash
python3 scripts/run_block_sweep.py \
  --elements 1048576 4194304 16777216 \
  --blocks 128 256 512 \
  --workload sgd-step \
  --output reports/block_sweep.csv
```

The same sweep is available through `make sweep` after the project is built. Passing more than one `--elements` value is useful for checking where launch overhead stops dominating and the memory-bound kernel reaches a steadier bandwidth regime.

## Timing note

The kernel-time metrics isolate device execution. The transfer and end-to-end fields show the cost of moving inputs and outputs across PCIe/NVLink, which is often the dominant cost for a bandwidth-bound vector operation like SAXPY.

## Project structure

```text
.
├── src/saxpy.cu       # host code, CUDA kernel, timing, and CLI
├── scripts/
│   └── run_block_sweep.py
├── CMakeLists.txt     # CUDA build configuration
├── Makefile           # build and run shortcuts
└── README.md
```
