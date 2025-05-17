#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint> // For int32_t
#include <cstring> // For memcpy
#include <CL/cl.h>
#include <cstdlib> // For std::atoi, std::atof
#include <cctype> // For std::tolower
#include <string> // For std::string

// Function to print help message
void print_help() {
    std::cout << "Usage: ./vector_mac -p <megapixels> -o <num_operations> -m <mode>\n"
              << "  -p <megapixels>    : Memory size in megapixels (e.g., 2 for 2 MP)\n"
              << "  -o <num_operations>: Number of operations (e.g., 25)\n"
              << "  -m <mode>          : MAC operation mode (S for sequential, P for parallel)\n"
              << "Example: ./vector_mac -p 2 -o 25 -m P\n";
}

int main(int argc, char* argv[]) {
    // Parse command-line flags
    double megapixels = 0.0;
    int num_operations = 0;
    char mode = '\0';
    bool p_set = false, o_set = false, m_set = false;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::cerr << "Error: Missing value for flag " << argv[i] << ".\n";
            print_help();
            return 1;
        }
        std::string flag = argv[i];
        std::string value = argv[i + 1];

        if (flag == "-p") {
            megapixels = std::atof(value.c_str());
            p_set = true;
        } else if (flag == "-o") {
            num_operations = std::atoi(value.c_str());
            o_set = true;
        } else if (flag == "-m") {
            mode = std::tolower(value[0]);
            m_set = true;
        } else {
            std::cerr << "Error: Unknown flag " << flag << ".\n";
            print_help();
            return 1;
        }
    }

    // Validate parsed arguments
    if (!p_set || !o_set || !m_set) {
        std::cerr << "Error: Missing required flags (-p, -o, -m).\n";
        print_help();
        return 1;
    }
    if (megapixels <= 0) {
        std::cerr << "Error: Megapixels must be a positive number.\n";
        print_help();
        return 1;
    }
    if (num_operations <= 0) {
        std::cerr << "Error: Number of operations must be a positive integer.\n";
        print_help();
        return 1;
    }
    if (mode != 's' && mode != 'p') {
        std::cerr << "Error: Mode must be 'S' or 'P'.\n";
        print_help();
        return 1;
    }
    bool is_parallel = (mode == 'p');

    // Calculate buffer size (pixels = megapixels * 1,000,000)
    size_t SIZE = static_cast<size_t>(megapixels * 1000000);
    unsigned int size_uint = static_cast<unsigned int>(SIZE); // For kernel argument

    // Calculate number of work-groups for parallel mode
    size_t localWorkSize = is_parallel ? 256 : 1;
    size_t globalWorkSize = is_parallel ? ((SIZE + localWorkSize - 1) / localWorkSize) * localWorkSize : 1;
    size_t numWorkGroups = is_parallel ? globalWorkSize / localWorkSize : 0;

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_seq, kernel_products, kernel_block_scan, kernel_block_combine;

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
    std::string kernel_file = is_parallel ? "vector_mac_par.cl" : "vector_mac_seq.cl";
    {
        std::ifstream file(kernel_file, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open kernel file: " << kernel_file << std::endl;
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

    // Create kernels based on mode
    if (is_parallel) {
        kernel_products = clCreateKernel(program, "vector_mac_products", &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create products kernel: " << err << std::endl;
            return 1;
        }
        kernel_block_scan = clCreateKernel(program, "vector_mac_block_scan", &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create block scan kernel: " << err << std::endl;
            return 1;
        }
        kernel_block_combine = clCreateKernel(program, "vector_mac_block_combine", &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create block combine kernel: " << err << std::endl;
            return 1;
        }
    } else {
        kernel_seq = clCreateKernel(program, "vector_mac", &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create sequential kernel: " << err << std::endl;
            return 1;
        }
    }

    // Create buffers for each operation using pinned memory
    std::vector<cl_mem> bufferA(num_operations), bufferB(num_operations), bufferC(num_operations);
    std::vector<cl_mem> bufferProducts(is_parallel ? num_operations : 0);
    std::vector<cl_mem> bufferBlockSums(is_parallel ? num_operations : 0);
    std::vector<std::vector<int32_t>> block_sums_cpu(is_parallel ? num_operations : 0);
    for (int op = 0; op < num_operations; ++op) {
        bufferA[op] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(int32_t) * SIZE, nullptr, &err);
        bufferB[op] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(int32_t) * SIZE, nullptr, &err);
        bufferC[op] = clCreateBuffer(context, is_parallel ? CL_MEM_READ_WRITE : CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(int32_t) * SIZE, nullptr, &err);
        if (is_parallel) {
            bufferProducts[op] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int32_t) * SIZE, nullptr, &err);
            bufferBlockSums[op] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int32_t) * numWorkGroups, nullptr, &err);
            block_sums_cpu[op].resize(numWorkGroups);
        }
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

    // Enqueue kernels based on mode
    for (int op = 0; op < num_operations; ++op) {
        if (is_parallel) {
            // Set arguments for products kernel
            clSetKernelArg(kernel_products, 0, sizeof(cl_mem), &bufferA[op]);
            clSetKernelArg(kernel_products, 1, sizeof(cl_mem), &bufferB[op]);
            clSetKernelArg(kernel_products, 2, sizeof(cl_mem), &bufferProducts[op]);
            clSetKernelArg(kernel_products, 3, sizeof(unsigned int), &size_uint);
            err = clEnqueueNDRangeKernel(queue, kernel_products, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to enqueue products kernel for operation " << op << ": " << err << std::endl;
                return 1;
            }

            // Set arguments for block scan kernel
            clSetKernelArg(kernel_block_scan, 0, sizeof(cl_mem), &bufferProducts[op]);
            clSetKernelArg(kernel_block_scan, 1, sizeof(cl_mem), &bufferC[op]);
            clSetKernelArg(kernel_block_scan, 2, sizeof(cl_mem), &bufferBlockSums[op]);
            clSetKernelArg(kernel_block_scan, 3, sizeof(unsigned int), &size_uint);
            clSetKernelArg(kernel_block_scan, 4, sizeof(int) * localWorkSize, nullptr); // Local memory for scratch
            err = clEnqueueNDRangeKernel(queue, kernel_block_scan, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to enqueue block scan kernel for operation " << op << ": " << err << std::endl;
                return 1;
            }

            // Read block sums to CPU
            void* ptrBlockSums = clEnqueueMapBuffer(queue, bufferBlockSums[op], CL_TRUE, CL_MAP_READ, 0, sizeof(int32_t) * numWorkGroups, 0, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to map block sums buffer for operation " << op << ": " << err << std::endl;
                return 1;
            }
            std::memcpy(block_sums_cpu[op].data(), ptrBlockSums, sizeof(int32_t) * numWorkGroups);
            clEnqueueUnmapMemObject(queue, bufferBlockSums[op], ptrBlockSums, 0, nullptr, nullptr);

            // Compute prefix sum of block sums on CPU
            for (size_t i = 1; i < numWorkGroups; ++i) {
                block_sums_cpu[op][i] += block_sums_cpu[op][i - 1];
            }

            // Write updated block sums back to GPU
            ptrBlockSums = clEnqueueMapBuffer(queue, bufferBlockSums[op], CL_TRUE, CL_MAP_WRITE, 0, sizeof(int32_t) * numWorkGroups, 0, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to map block sums buffer for operation " << op << ": " << err << std::endl;
                return 1;
            }
            std::memcpy(ptrBlockSums, block_sums_cpu[op].data(), sizeof(int32_t) * numWorkGroups);
            clEnqueueUnmapMemObject(queue, bufferBlockSums[op], ptrBlockSums, 0, nullptr, nullptr);

            // Set arguments for block combine kernel
            clSetKernelArg(kernel_block_combine, 0, sizeof(cl_mem), &bufferC[op]);
            clSetKernelArg(kernel_block_combine, 1, sizeof(cl_mem), &bufferBlockSums[op]);
            clSetKernelArg(kernel_block_combine, 2, sizeof(unsigned int), &size_uint);
            err = clEnqueueNDRangeKernel(queue, kernel_block_combine, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to enqueue block combine kernel for operation " << op << ": " << err << std::endl;
                return 1;
            }
        } else {
            // Set arguments for sequential kernel
            clSetKernelArg(kernel_seq, 0, sizeof(cl_mem), &bufferA[op]);
            clSetKernelArg(kernel_seq, 1, sizeof(cl_mem), &bufferB[op]);
            clSetKernelArg(kernel_seq, 2, sizeof(cl_mem), &bufferC[op]);
            clSetKernelArg(kernel_seq, 3, sizeof(unsigned int), &size_uint);
            err = clEnqueueNDRangeKernel(queue, kernel_seq, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed to enqueue sequential kernel for operation " << op << ": " << err << std::endl;
                return 1;
            }
        }
    }

    // Finish all kernel executions
    clFinish(queue);

    // End timing GPU MAC computation
    auto gpu_mac_end = std::chrono::high_resolution_clock::now();
    auto gpu_mac_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_mac_end - gpu_mac_start);
    std::cout << (is_parallel ? "GPU Parallel MAC Computation Time (" : "GPU Sequential MAC Computation Time (") 
              << num_operations << " operations): " << gpu_mac_duration.count() / 1000.0 << " ms" << std::endl;

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
        if (is_parallel) {
            clReleaseMemObject(bufferProducts[op]);
            clReleaseMemObject(bufferBlockSums[op]);
        }
    }
    if (is_parallel) {
        clReleaseKernel(kernel_products);
        clReleaseKernel(kernel_block_scan);
        clReleaseKernel(kernel_block_combine);
    } else {
        clReleaseKernel(kernel_seq);
    }
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}