__kernel void vector_mac(__global const int *a, __global const int *b, __global int *c, const unsigned int size) {
    int acc = 0;
    for (unsigned int i = 0; i < size; ++i) {
        acc += a[i] * b[i];
        c[i] = acc;
    }
}