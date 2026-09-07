#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& message) : std::runtime_error(message) {}
};

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        throw CudaError(std::string(expression) + ": " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression)

struct Options {
    std::size_t elements = 1 << 24;
    int iterations = 100;
    int warmup_iterations = 1;
    int block_size = 256;
    float alpha = 2.0F;
    float learning_rate = 0.01F;
    std::uint32_t seed = 7;
    std::string workload = "saxpy";
    std::string json_output;
    std::string csv_output;
};

struct GpuTiming {
    float host_to_device_ms = 0.0F;
    float kernel_ms = 0.0F;
    float device_to_host_ms = 0.0F;
    float end_to_end_ms = 0.0F;
};

template <typename T>
T parse_value(const std::string& value, const std::string& name);

template <>
std::size_t parse_value(const std::string& value, const std::string& name) {
    try {
        return std::stoull(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
}

template <>
int parse_value(const std::string& value, const std::string& name) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
}

template <>
float parse_value(const std::string& value, const std::string& name) {
    try {
        return std::stof(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + name + ": " + value);
    }
}

std::uint32_t parse_seed(const std::string& value, const std::string& name) {
    const auto parsed = parse_value<std::size_t>(value, name);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("seed must fit in uint32_t: " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            std::cout << "Usage: cuda_saxpy [options]\n"
                      << "  --elements N       number of vector elements\n"
                      << "  --iterations N     timed repetitions\n"
                      << "  --warmup N         untimed kernel repetitions\n"
                      << "  --block-size N     CUDA threads per block\n"
                      << "  --alpha VALUE      SAXPY scaling factor\n"
                      << "  --workload NAME    saxpy or sgd-step\n"
                      << "  --learning-rate V  step size used by --workload sgd-step\n"
                      << "  --seed N           random input seed\n"
                      << "  --json-output PATH write machine-readable JSON results\n"
                      << "  --csv-output PATH  append one benchmark row to a CSV file\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--elements") {
            options.elements = parse_value<std::size_t>(value, argument);
        } else if (argument == "--iterations") {
            options.iterations = parse_value<int>(value, argument);
        } else if (argument == "--warmup") {
            options.warmup_iterations = parse_value<int>(value, argument);
        } else if (argument == "--block-size") {
            options.block_size = parse_value<int>(value, argument);
        } else if (argument == "--alpha") {
            options.alpha = parse_value<float>(value, argument);
        } else if (argument == "--workload") {
            options.workload = value;
        } else if (argument == "--learning-rate") {
            options.learning_rate = parse_value<float>(value, argument);
        } else if (argument == "--seed") {
            options.seed = parse_seed(value, argument);
        } else if (argument == "--json-output") {
            options.json_output = value;
        } else if (argument == "--csv-output") {
            options.csv_output = value;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.elements == 0 || options.iterations < 1 || options.warmup_iterations < 0 ||
        options.block_size < 1 ||
        options.block_size > 1024) {
        throw std::invalid_argument("elements, iterations, and block size must be positive; warmup must not be negative; block size must be <= 1024");
    }
    if (options.workload != "saxpy" && options.workload != "sgd-step") {
        throw std::invalid_argument("workload must be either saxpy or sgd-step");
    }
    if (!std::isfinite(options.alpha) || !std::isfinite(options.learning_rate) ||
        options.learning_rate <= 0.0F) {
        throw std::invalid_argument("alpha must be finite and learning-rate must be positive");
    }
    return options;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    T* data() { return data_; }

private:
    T* data_ = nullptr;
    std::size_t count_;
};

__global__ void saxpy_kernel(
    const float* x,
    const float* y,
    float* output,
    float alpha,
    std::size_t elements) {
    const std::size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t index = start; index < elements; index += stride) {
        output[index] = alpha * x[index] + y[index];
    }
}

void saxpy_cpu(
    const std::vector<float>& x,
    const std::vector<float>& y,
    std::vector<float>& output,
    float alpha) {
    for (std::size_t index = 0; index < x.size(); ++index) {
        output[index] = alpha * x[index] + y[index];
    }
}

double benchmark_cpu(
    const std::vector<float>& x,
    const std::vector<float>& y,
    std::vector<float>& output,
    float alpha,
    int iterations) {
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        saxpy_cpu(x, y, output, alpha);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count() / iterations;
}

std::size_t launch_block_count(std::size_t elements, int block_size, int multiprocessor_count) {
    const auto required_blocks = (elements + block_size - 1) / block_size;
    const auto occupancy_blocks = static_cast<std::size_t>(multiprocessor_count * 32);
    return std::min<std::size_t>(required_blocks, occupancy_blocks);
}

void validate_device_launch_options(const Options& options, const cudaDeviceProp& properties) {
    if (options.block_size > properties.maxThreadsPerBlock) {
        throw std::invalid_argument(
            "block size exceeds this device limit of " +
            std::to_string(properties.maxThreadsPerBlock));
    }
    const auto blocks = launch_block_count(
        options.elements,
        options.block_size,
        properties.multiProcessorCount);
    if (blocks == 0 || blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::invalid_argument("computed grid size is not valid for this device");
    }
}

float event_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    return elapsed;
}

GpuTiming benchmark_gpu(
    const std::vector<float>& x,
    const std::vector<float>& y,
    std::vector<float>& output,
    float alpha,
    int block_size,
    int iterations,
    int warmup_iterations,
    const cudaDeviceProp& properties) {
    DeviceBuffer<float> device_x(x.size());
    DeviceBuffer<float> device_y(y.size());
    DeviceBuffer<float> device_output(output.size());

    const auto block_count = launch_block_count(
        x.size(),
        block_size,
        properties.multiProcessorCount);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(device_x.data(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_y.data(), y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    const float h2d_ms = event_elapsed_ms(start, stop);

    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        saxpy_kernel<<<static_cast<unsigned int>(block_count), block_size>>>(
            device_x.data(),
            device_y.data(),
            device_output.data(),
            alpha,
            x.size());
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        saxpy_kernel<<<static_cast<unsigned int>(block_count), block_size>>>(
            device_x.data(),
            device_y.data(),
            device_output.data(),
            alpha,
            x.size());
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float kernel_ms = event_elapsed_ms(start, stop) / iterations;

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.data(), output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    const float d2h_ms = event_elapsed_ms(start, stop);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return GpuTiming{
        h2d_ms,
        kernel_ms,
        d2h_ms,
        h2d_ms + kernel_ms + d2h_ms,
    };
}

float maximum_error(const std::vector<float>& expected, const std::vector<float>& actual) {
    float error = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        error = std::max(error, std::abs(expected[index] - actual[index]));
    }
    return error;
}

double effective_bandwidth_gbps(std::size_t elements, float elapsed_ms) {
    if (elapsed_ms <= 0.0F) {
        return 0.0;
    }
    const double bytes = 3.0 * static_cast<double>(elements) * sizeof(float);
    return bytes / (static_cast<double>(elapsed_ms) * 1'000'000.0);
}

double saxpy_gflops(std::size_t elements, float elapsed_ms) {
    if (elapsed_ms <= 0.0F) {
        return 0.0;
    }
    const double operations = 2.0 * static_cast<double>(elements);
    return operations / (static_cast<double>(elapsed_ms) * 1'000'000.0);
}

double arithmetic_intensity_flop_per_byte() {
    return 2.0 / (3.0 * static_cast<double>(sizeof(float)));
}

float operation_alpha(const Options& options) {
    return options.workload == "sgd-step" ? -options.learning_rate : options.alpha;
}

double bytes_to_mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double device_working_set_mib(std::size_t elements) {
    return bytes_to_mib(3 * elements * sizeof(float));
}

double host_working_set_mib(std::size_t elements) {
    return bytes_to_mib(4 * elements * sizeof(float));
}

double launched_threads_per_sm(
    std::size_t kernel_blocks,
    int block_size,
    int multiprocessor_count) {
    if (multiprocessor_count <= 0) {
        return 0.0;
    }
    return static_cast<double>(kernel_blocks * static_cast<std::size_t>(block_size)) /
        static_cast<double>(multiprocessor_count);
}

double transfer_bandwidth_gbps(std::size_t elements, float h2d_ms, float d2h_ms) {
    const double elapsed_ms = static_cast<double>(h2d_ms + d2h_ms);
    if (elapsed_ms <= 0.0) {
        return 0.0;
    }
    const double bytes = 3.0 * static_cast<double>(elements) * sizeof(float);
    return bytes / (elapsed_ms * 1'000'000.0);
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    for (const char character : value) {
        if (character == '"' || character == '\\') {
            escaped += '\\';
        }
        escaped += character;
    }
    return escaped;
}

void write_json_report(
    const Options& options,
    const cudaDeviceProp& properties,
    std::size_t kernel_blocks,
    double cpu_ms,
    const GpuTiming& gpu_timing,
    double bandwidth_gbps,
    double transfer_bandwidth,
    double gflops,
    double arithmetic_intensity,
    double kernel_speedup,
    double end_to_end_speedup,
    float error) {
    const std::filesystem::path output_path(options.json_output);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream report(output_path);
    if (!report) {
        throw std::runtime_error("could not open JSON output: " + options.json_output);
    }
    report << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"gpu\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"workload\": \"" << json_escape(options.workload) << "\",\n"
           << "  \"elements\": " << options.elements << ",\n"
           << "  \"block_size\": " << options.block_size << ",\n"
           << "  \"kernel_blocks\": " << kernel_blocks << ",\n"
           << "  \"warmup_iterations\": " << options.warmup_iterations << ",\n"
           << "  \"iterations\": " << options.iterations << ",\n"
           << "  \"alpha\": " << options.alpha << ",\n"
           << "  \"learning_rate\": " << options.learning_rate << ",\n"
           << "  \"effective_alpha\": " << operation_alpha(options) << ",\n"
           << "  \"host_working_set_mib\": " << host_working_set_mib(options.elements) << ",\n"
           << "  \"device_working_set_mib\": " << device_working_set_mib(options.elements) << ",\n"
           << "  \"launched_threads_per_sm\": "
           << launched_threads_per_sm(kernel_blocks, options.block_size, properties.multiProcessorCount) << ",\n"
           << "  \"cpu_ms\": " << cpu_ms << ",\n"
           << "  \"gpu_h2d_ms\": " << gpu_timing.host_to_device_ms << ",\n"
           << "  \"gpu_kernel_ms\": " << gpu_timing.kernel_ms << ",\n"
           << "  \"gpu_d2h_ms\": " << gpu_timing.device_to_host_ms << ",\n"
           << "  \"gpu_end_to_end_ms\": " << gpu_timing.end_to_end_ms << ",\n"
           << "  \"effective_bandwidth_gbps\": " << bandwidth_gbps << ",\n"
           << "  \"transfer_bandwidth_gbps\": " << transfer_bandwidth << ",\n"
           << "  \"gflops\": " << gflops << ",\n"
           << "  \"arithmetic_intensity_flop_per_byte\": " << arithmetic_intensity << ",\n"
           << "  \"kernel_speedup\": " << kernel_speedup << ",\n"
           << "  \"end_to_end_speedup\": " << end_to_end_speedup << ",\n"
           << "  \"maximum_absolute_error\": " << error << "\n"
           << "}\n";
}

void write_csv_report(
    const Options& options,
    const cudaDeviceProp& properties,
    std::size_t kernel_blocks,
    double cpu_ms,
    const GpuTiming& gpu_timing,
    double bandwidth_gbps,
    double transfer_bandwidth,
    double gflops,
    double arithmetic_intensity,
    double kernel_speedup,
    double end_to_end_speedup,
    float error) {
    const std::filesystem::path output_path(options.csv_output);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    const bool write_header = !std::filesystem::exists(output_path) ||
        std::filesystem::file_size(output_path) == 0;
    std::ofstream report(output_path, std::ios::app);
    if (!report) {
        throw std::runtime_error("could not open CSV output: " + options.csv_output);
    }
    if (write_header) {
        report << "gpu,elements,block_size,kernel_blocks,warmup_iterations,iterations,alpha,"
               << "workload,learning_rate,effective_alpha,"
               << "host_working_set_mib,device_working_set_mib,launched_threads_per_sm,"
               << "cpu_ms,gpu_h2d_ms,gpu_kernel_ms,gpu_d2h_ms,gpu_end_to_end_ms,"
               << "effective_bandwidth_gbps,transfer_bandwidth_gbps,gflops,"
               << "arithmetic_intensity_flop_per_byte,kernel_speedup,end_to_end_speedup,"
               << "maximum_absolute_error\n";
    }
    report << std::fixed << std::setprecision(6)
           << '"' << json_escape(properties.name) << '"' << ','
           << options.elements << ','
           << options.block_size << ','
           << kernel_blocks << ','
           << options.warmup_iterations << ','
           << options.iterations << ','
           << options.alpha << ','
           << options.workload << ','
           << options.learning_rate << ','
           << operation_alpha(options) << ','
           << host_working_set_mib(options.elements) << ','
           << device_working_set_mib(options.elements) << ','
           << launched_threads_per_sm(kernel_blocks, options.block_size, properties.multiProcessorCount) << ','
           << cpu_ms << ','
           << gpu_timing.host_to_device_ms << ','
           << gpu_timing.kernel_ms << ','
           << gpu_timing.device_to_host_ms << ','
           << gpu_timing.end_to_end_ms << ','
           << bandwidth_gbps << ','
           << transfer_bandwidth << ','
           << gflops << ','
           << arithmetic_intensity << ','
           << kernel_speedup << ','
           << end_to_end_speedup << ','
           << error << '\n';
}

}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        validate_device_launch_options(options, properties);

        std::mt19937 generator(options.seed);
        std::normal_distribution<float> distribution(0.0F, 1.0F);
        std::vector<float> x(options.elements);
        std::vector<float> y(options.elements);
        std::vector<float> cpu_output(options.elements);
        std::vector<float> gpu_output(options.elements);
        for (std::size_t index = 0; index < options.elements; ++index) {
            x[index] = distribution(generator);
            y[index] = distribution(generator);
        }

        const float effective_alpha = operation_alpha(options);
        const double cpu_ms = benchmark_cpu(
            x, y, cpu_output, effective_alpha, options.iterations);
        const GpuTiming gpu_timing = benchmark_gpu(
            x,
            y,
            gpu_output,
            effective_alpha,
            options.block_size,
            options.iterations,
            options.warmup_iterations,
            properties);
        const float error = maximum_error(cpu_output, gpu_output);
        const double bandwidth_gbps = effective_bandwidth_gbps(options.elements, gpu_timing.kernel_ms);
        const double transfer_gbps = transfer_bandwidth_gbps(
            options.elements,
            gpu_timing.host_to_device_ms,
            gpu_timing.device_to_host_ms);
        const double gflops = saxpy_gflops(options.elements, gpu_timing.kernel_ms);
        const double arithmetic_intensity = arithmetic_intensity_flop_per_byte();
        const double kernel_speedup = gpu_timing.kernel_ms > 0.0F ? cpu_ms / gpu_timing.kernel_ms : 0.0;
        const double end_to_end_speedup = gpu_timing.end_to_end_ms > 0.0F ? cpu_ms / gpu_timing.end_to_end_ms : 0.0;

        const auto kernel_blocks = launch_block_count(
            options.elements,
            options.block_size,
            properties.multiProcessorCount);
        std::cout << std::fixed << std::setprecision(3)
                  << "GPU: " << properties.name << "\n"
                  << "Workload: " << options.workload << "\n"
                  << "Elements: " << options.elements << "\n"
                  << "Block size: " << options.block_size << "\n"
                  << "Kernel blocks: " << kernel_blocks << "\n"
                  << "Effective alpha: " << effective_alpha << "\n"
                  << "Host working set: " << host_working_set_mib(options.elements) << " MiB\n"
                  << "Device working set: " << device_working_set_mib(options.elements) << " MiB\n"
                  << "Launched threads per SM: "
                  << launched_threads_per_sm(kernel_blocks, options.block_size, properties.multiProcessorCount) << "\n"
                  << "Warm-up iterations: " << options.warmup_iterations << "\n"
                  << "CPU average: " << cpu_ms << " ms\n"
                  << "GPU H2D copy: " << gpu_timing.host_to_device_ms << " ms\n"
                  << "GPU kernel average: " << gpu_timing.kernel_ms << " ms\n"
                  << "GPU D2H copy: " << gpu_timing.device_to_host_ms << " ms\n"
                  << "GPU end-to-end: " << gpu_timing.end_to_end_ms << " ms\n"
                  << "GPU effective bandwidth: " << bandwidth_gbps << " GB/s\n"
                  << "Transfer bandwidth: " << transfer_gbps << " GB/s\n"
                  << "Achieved throughput: " << gflops << " GFLOP/s\n"
                  << "Arithmetic intensity: " << arithmetic_intensity << " FLOP/byte\n"
                  << "Kernel speedup: " << kernel_speedup << "x\n"
                  << "End-to-end speedup: " << end_to_end_speedup << "x\n"
                  << "Maximum absolute error: " << error << "\n";
        if (!options.json_output.empty()) {
            write_json_report(
                options,
                properties,
                kernel_blocks,
                cpu_ms,
                gpu_timing,
                bandwidth_gbps,
                transfer_gbps,
                gflops,
                arithmetic_intensity,
                kernel_speedup,
                end_to_end_speedup,
                error);
        }
        if (!options.csv_output.empty()) {
            write_csv_report(
                options,
                properties,
                kernel_blocks,
                cpu_ms,
                gpu_timing,
                bandwidth_gbps,
                transfer_gbps,
                gflops,
                arithmetic_intensity,
                kernel_speedup,
                end_to_end_speedup,
                error);
        }
        return error < 1e-5F ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
