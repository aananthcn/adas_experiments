#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint> // For int32_t
#include <cstring> // For memcpy
#include <CL/cl.h>

int main() {
    // Prompt for megapixels and number of operations
    double megapixels;
    int num_operations;
    std::cout << "Enter memory size in megapixels (e.g., 2 for 2 MP): ";
    std::cin >> megapixels;
    std::cout << "Enter number of operations (e.g., 25): ";
    std::cin >> num_operations;

    // Calculate buffer size (pixels = megapixels * 1,000,000)
    size_t SIZE = static_cast<size_t>(megapixels * 1000000);
    unsigned int size_uint = static_cast<unsigned int>(SIZE); // For kernel argument

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Start timing CPU initialization
    auto init_start = std::chrono::high_resolution_clock::now();

    // Initialize input vectors for each operation
    std::vector<std::vector<int32_t>> a(num_operations, std::vector<int32_t>(SIZE));
    std::vector<std::vector<int32_t>> b(num_operations, std::vector<int32_t>(SIZE));
    std::vector<std::vector<int32_t>> c(num_operations, std::vector<int32_t>(SIZE));
    std::vector<std::vector<int32_t>> c_cpu(num_operations, std::vector<int32_t>(SIZE));
    for (int op = 0; op < num_operations; ++op) {
        for (size_t i = 0; i < SIZE; ++i) {
            a[op][i] = static_cast<int32_t>(i + op); // Vary data slightly per operation
            b[op][i] = static_cast<int32_t>(SIZE - i - op);
        }
    }

    // End timing CPU initialization
    auto init_end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start);
    std::cout << "CPU Initialization Time: " << init_duration.count() / 1000.0 << " ms" << std::endl;

    // Start timing CPU MAC
    auto cpu_mac_start = std::chrono::high_resolution_clock::now();

    // Perform multiply-and-accumulate on CPU for each operation
    for (int op = 0; op < num_operations; ++op) {
        int32_t acc = 0;
        for (size_t i = 0; i < SIZE; ++i) {
            acc += a[op][i] * b[op][i];
            c_cpu[op][i] = acc;
        }
    }

    // End timing CPU MAC
    auto cpu_mac_end = std::chrono::high_resolution_clock::now();
    auto cpu_mac_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_mac_end - cpu_mac_start);
    std::cout << "CPU MAC Time: " << cpu_mac_duration.count() / 1000.0 << " ms" << std::endl;

    // Get platform and GPU device
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform: " << err << std::endl;
        return 1;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get GPU device: " << err << std::endl;
        return 1;
    }

    // Create context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context: " << err << std::endl;
        return 1;
    }

    // Create command queue with out-of-order execution
    queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue: " << err << std::endl;
        return 1;
    }

    // Read and create program
    std::string kernelSource;
    {
        std::ifstream file("vector_mac.cl", std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open kernel file: vector_mac.cl" << std::endl;
            return 1;
        }
        kernelSource = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }
    const char* source = kernelSource.c_str();
    program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program: " << err << std::endl;
        return 1;
    }

    // Build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log.data() << std::endl;
        return 1;
    }

    // Create kernel
    kernel = clCreateKernel(program, "vector_mac", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel: " << err << std::endl;
        return 1;
    }

    // Create buffers for each operation using pinned memory
    std::vector<cl_mem> bufferA(num_operations), bufferB(num_operations), bufferC(num_operations);
    for (int op = 0; op < num_operations; ++op) {
        bufferA[op] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(int32_t) * SIZE, nullptr, &err);
        bufferB[op] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(int32_t) * SIZE, nullptr, &err);
        bufferC[op] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(int32_t) * SIZE, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create buffers for operation " << op << ": " << err << std::endl;
            return 1;
        }
    }

    // Start timing GPU data transfer (write)
    auto gpu_transfer_write_start = std::chrono::high_resolution_clock::now();

    // Write data to pinned buffers
    for (int op = 0; op < num_operations; ++op) {
        void* ptrA = clEnqueueMapBuffer(queue, bufferA[op], CL_TRUE, CL_MAP_WRITE, 0, sizeof(int32_t) * SIZE, 0, nullptr, nullptr, &err);
        void* ptrB = clEnqueueMapBuffer(queue, bufferB[op], CL_TRUE, CL_MAP_WRITE, 0, sizeof(int32_t) * SIZE, 0, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to map buffers for operation " << op << ": " << err << std::endl;
            return 1;
        }
        std::memcpy(ptrA, a[op].data(), sizeof(int32_t) * SIZE);
        std::memcpy(ptrB, b[op].data(), sizeof(int32_t) * SIZE);
        clEnqueueUnmapMemObject(queue, bufferA[op], ptrA, 0, nullptr, nullptr);
        clEnqueueUnmapMemObject(queue, bufferB[op], ptrB, 0, nullptr, nullptr);
    }

    // End timing GPU data transfer (write)
    auto gpu_transfer_write_end = std::chrono::high_resolution_clock::now();
    auto gpu_transfer_write_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_transfer_write_end - gpu_transfer_write_start);
    std::cout << "GPU Data Transfer Time (Write): " << gpu_transfer_write_duration.count() / 1000.0 << " ms" << std::endl;

    // Start timing GPU MAC computation
    auto gpu_mac_start = std::chrono::high_resolution_clock::now();

    // Enqueue all kernel executions (single work-item per operation)
    size_t globalWorkSize = 1; // Single work-item
    size_t localWorkSize = 1;
    for (int op = 0; op < num_operations; ++op) {
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA[op]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB[op]);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC[op]);
        clSetKernelArg(kernel, 3, sizeof(unsigned int), &size_uint);
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to enqueue kernel for operation " << op << ": " << err << std::endl;
            return 1;
        }
    }

    // Finish all kernel executions
    clFinish(queue);

    // End timing GPU MAC computation
    auto gpu_mac_end = std::chrono::high_resolution_clock::now();
    auto gpu_mac_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_mac_end - gpu_mac_start);
    std::cout << "GPU MAC Computation Time (" << num_operations << " operations): " << gpu_mac_duration.count() / 1000.0 << " ms" << std::endl;

    // Start timing GPU data transfer (read)
    auto gpu_transfer_read_start = std::chrono::high_resolution_clock::now();

    // Read results for all operations
    for (int op = 0; op < num_operations; ++op) {
        void* ptrC = clEnqueueMapBuffer(queue, bufferC[op], CL_TRUE, CL_MAP_READ, 0, sizeof(int32_t) * SIZE, 0, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to map buffer C for operation " << op << ": " << err << std::endl;
            return 1;
        }
        std::memcpy(c[op].data(), ptrC, sizeof(int32_t) * SIZE);
        clEnqueueUnmapMemObject(queue, bufferC[op], ptrC, 0, nullptr, nullptr);
    }

    // End timing GPU data transfer (read)
    auto gpu_transfer_read_end = std::chrono::high_resolution_clock::now();
    auto gpu_transfer_read_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_transfer_read_end - gpu_transfer_read_start);
    std::cout << "GPU Data Transfer Time (Read): " << gpu_transfer_read_duration.count() / 1000.0 << " ms" << std::endl;

    // Start timing CPU verification
    auto verify_start = std::chrono::high_resolution_clock::now();

    // Verify results
    bool passed = true;
    for (int op = 0; op < num_operations; ++op) {
        for (size_t i = 0; i < SIZE; ++i) {
            if (c[op][i] != c_cpu[op][i]) {
                std::cerr << "Verification failed at operation " << op << ", index " << i << ": GPU " << c[op][i] << " != CPU " << c_cpu[op][i] << std::endl;
                passed = false;
                break;
            }
        }
        if (!passed) break;
    }
    if (passed) {
        std::cout << "Verification passed!" << std::endl;
    }

    // End timing CPU verification
    auto verify_end = std::chrono::high_resolution_clock::now();
    auto verify_duration = std::chrono::duration_cast<std::chrono::microseconds>(verify_end - verify_start);
    std::cout << "CPU Verification Time: " << verify_duration.count() / 1000.0 << " ms" << std::endl;

    // Cleanup
    for (int op = 0; op < num_operations; ++op) {
        clReleaseMemObject(bufferA[op]);
        clReleaseMemObject(bufferB[op]);
        clReleaseMemObject(bufferC[op]);
    }
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}