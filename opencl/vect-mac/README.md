# Introduction
The program "vect_mac" stands for vector multiply and accumulate operation, done on both CPU and GPU to see how fast both does.

# The Design
The basic algorithm behnd this is, it creates 3 vectors a, b and c; then initialize a from 0 to SIZE_VECT-1 and initialize b with SIZE_VECT-1 to 0 (in reverse order) and then do the following:

 * c[i] = c[i-1] + a[i] * b[i]

The above operation will be done in both CPU and GPU and later the time and values will be verified to see how both performs.

## Sequential Mode
In sequential mode, the operation is done how a human would do. This would not effectively utilize the 1000s of cuda cores available in GPU. So the CPU is expected to do faster.

## Parallel Mode
In parallel mode, the algorithm makes best effort to do things in parallel so that more cores are utilized. The expectation is that the time of execution on GPU must be less than CPU, in this case.

# Program invocation
This program comes with CMake script to generate executable and once configured and built, the program can be invoked as below:

```
Usage: ./vector_mac -p <megapixels> -o <num_operations> -m <mode>
  -p <megapixels>    : Memory size in megapixels (e.g., 2 for 2 MP)
  -o <num_operations>: Number of operations (e.g., 25)
  -m <mode>          : MAC operation mode (S for sequential, P for parallel)
```

Example: ./vector_mac -p 2 -o 25 -m P