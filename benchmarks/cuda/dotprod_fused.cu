#include <stdio.h>
#include <time.h>

/*
 * Consolidated version incorporating:
 *   - map_2kernel:  __restrict__ on all pointers + grid-stride loop
 *   - reduce:       single-pass, fused two-level reduction (no second
 *                    kernel launch, no cudaMemcpy of partials back to host)
 *   - no atomics on the float being reduced (stays generic over any f,
 *     not just `+`); the only atomic is an int counter used purely to
 *     detect "last block to finish"
 *   - no bit shifts (plain integer division by 2, compiler
 *     strength-reduces this back to a shift automatically)
 *   - no volatile / warp-synchronous assumptions: every reduction level,
 *     including the last warp, goes through __syncthreads(), so the
 *     kernel is correct on Volta+ independent-thread-scheduling hardware
 *     without relying on undefined-behavior lockstep assumptions
 *   - identity element for the reduction is now a parameter instead of
 *     a hardcoded 0.0f, since 0.0f is only the correct identity for `+`;
 *     for `f = max` you'd pass -INFINITY, for `f = min` you'd pass
 *     +INFINITY, etc. This keeps the kernel genuinely generic over f.
 */

// ---------------------------------------------------------------------
// map: a3[i] = f(a1[i], a2[i])
// ---------------------------------------------------------------------
__global__
void map_2kernel(const float* __restrict__ a1,
                  const float* __restrict__ a2,
                  float* __restrict__ a3,
                  int size, float (*f)(float,float))
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid < size; tid += blockDim.x * gridDim.x)
        a3[tid] = f(a1[tid], a2[tid]);
}

// ---------------------------------------------------------------------
// reduce: single kernel launch, generic over f and over the identity.
// `partials` is scratch space of length == gridDim.x (numberOfBlocks),
// allocated by the caller. `out` receives the single final value.
// `reduction_counter` must be a device int initialized to 0 before the
// launch (and is reset to 0 at the end of the kernel so it's ready for
// the next launch without an extra host-side reset).
// ---------------------------------------------------------------------
__global__
void reduce_kernel_fused(const float* __restrict__ a,
                          float* __restrict__ partials,
                          float* __restrict__ out,
                          float (*f)(float,float),
                          float identity,
                          unsigned int* reduction_counter,
                          int n)
{
    __shared__ float cache[256];
    int cacheIndex = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // --- level 1: each block reduces its slice of `a` ---
    float temp = identity;
    for (; tid < n; tid += blockDim.x * gridDim.x)
        temp = f(a[tid], temp);

    cache[cacheIndex] = temp;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i = i / 2) {
        if (cacheIndex < i)
            cache[cacheIndex] = f(cache[cacheIndex + i], cache[cacheIndex]);
        __syncthreads();
    }

    __shared__ bool isLastBlock;
    if (cacheIndex == 0) {
        partials[blockIdx.x] = cache[0];
        __threadfence();  // make this block's write visible before it's counted

        unsigned int ticket = atomicAdd(reduction_counter, 1);
        isLastBlock = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    // --- level 2: only the last block to arrive reduces `partials` ---
    if (isLastBlock) {
        float t = identity;
        for (int i = cacheIndex; i < gridDim.x; i += blockDim.x)
            t = f(partials[i], t);
        cache[cacheIndex] = t;
        __syncthreads();

        for (int i = blockDim.x / 2; i > 0; i = i / 2) {
            if (cacheIndex < i)
                cache[cacheIndex] = f(cache[cacheIndex + i], cache[cacheIndex]);
            __syncthreads();
        }

        if (cacheIndex == 0) {
            *out = cache[0];
            *reduction_counter = 0;  // reset for next launch
        }
    }
}

//#############################

__device__
float anonymous_mult(float a, float b) { return a * b; }

__device__ void* anonymous_mult_ptr = (void*) anonymous_mult;

extern "C" void* get_anonymous_mult_ptr()
{
    void* host_function_ptr;
    cudaMemcpyFromSymbol(&host_function_ptr, anonymous_mult_ptr, sizeof(void*));
    return host_function_ptr;
}

//#############################

__device__
float anonymous_sum(float a, float b) { return a + b; }

__device__ void* anonymous_sum_ptr = (void*) anonymous_sum;

extern "C" void* get_anonymous_sum_ptr()
{
    void* host_function_ptr;
    cudaMemcpyFromSymbol(&host_function_ptr, anonymous_sum_ptr, sizeof(void*));
    return host_function_ptr;
}

//#############################

int main(int argc, char *argv[])
{
    float *a, *b, *final;
    float *dev_a, *dev_b, *dev_resp, *d_partials, *d_final;
    unsigned int *d_counter;
    cudaError_t j_error;

    int N = atoi(argv[1]);

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    final = (float*)malloc(sizeof(float));

    for (int i = 0; i < N; i++) a[i] = rand();
    for (int i = 0; i < N; i++) b[i] = rand();

    int threadsPerBlock = 256;
    int numberOfBlocks  = 64;  // fixed, occupancy-sized grid; not tied to N

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void**)&dev_a, N * sizeof(float));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMalloc((void**)&dev_resp, N * sizeof(float));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMalloc((void**)&d_partials, numberOfBlocks * sizeof(float));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMalloc((void**)&d_final, sizeof(float));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMalloc((void**)&d_counter, sizeof(unsigned int));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMemset(d_counter, 0, sizeof(unsigned int));

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    float (*f1)(float,float) = (float (*)(float,float)) get_anonymous_mult_ptr();
    float (*f2)(float,float) = (float (*)(float,float)) get_anonymous_sum_ptr();

    map_2kernel<<<numberOfBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_resp, N, f1);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    // identity for `+` is 0.0f; pass -INFINITY for max, +INFINITY for min, etc.
    reduce_kernel_fused<<<numberOfBlocks, threadsPerBlock>>>(
        dev_resp, d_partials, d_final, f2, 0.0f, d_counter, N);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    cudaMemcpy(final, d_final, sizeof(float), cudaMemcpyDeviceToHost);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_resp);
    cudaFree(d_partials);
    cudaFree(d_final);
    cudaFree(d_counter);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("CUDA\t%d\t%3.1f\n", N, time);

    free(a);
    free(b);
    free(final);

    return 0;
}
