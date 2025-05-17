// Kernel to compute products a[i] * b[i] in parallel
__kernel void vector_mac_products(__global const int *a, __global const int *b, __global int *products, const unsigned int size) {
    unsigned int gid = get_global_id(0);
    if (gid < size) {
        products[gid] = a[gid] * b[gid];
    }
}

// Kernel to compute inclusive block-level prefix sums
__kernel void vector_mac_block_scan(__global const int *products, __global int *c, __global int *block_sums, const unsigned int size, __local int *scratch) {
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_size = get_local_size(0);

    // Load data into local memory
    int value = 0;
    if (gid < size) {
        value = products[gid];
    }
    scratch[lid] = value;

    // Synchronize to ensure all local memory writes are complete
    barrier(CLK_LOCAL_MEM_FENCE);

    // Inclusive prefix sum within work-group
    for (unsigned int offset = 1; offset < group_size; offset *= 2) {
        int temp = 0;
        if (lid >= offset) {
            temp = scratch[lid - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= offset) {
            scratch[lid] += temp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result to global memory
    if (gid < size) {
        c[gid] = scratch[lid];
    }

    // Write block sum to block_sums
    if (lid == group_size - 1 && gid < size) {
        block_sums[group_id] = scratch[lid];
    }
}

// Kernel to combine block sums with block-level prefix sums
__kernel void vector_mac_block_combine(__global int *c, __global const int *block_sums, const unsigned int size) {
    unsigned int gid = get_global_id(0);
    unsigned int group_id = get_group_id(0);

    if (gid < size && group_id > 0) {
        c[gid] += block_sums[group_id - 1];
    }
}