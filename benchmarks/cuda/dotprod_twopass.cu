#include <stdio.h>
#include <time.h>

/*
 * Two-kernel-launch version of the reduction (as opposed to the fused,
 * single-launch version), for direct comparison. Everything else
 * discussed still applies:
 *   - map_2kernel: __restrict__ via local aliases, while loop
 *   - reduce: generic over f via an explicit identity parameter,
 *     no atomics on the float being reduced at all (not even an int
 *     counter here - the two separate kernel launches on the same
 *     stream already guarantee ordering, so there's nothing to detect)
 *   - while loops instead of for, no bit shifts
 *
 * reduce_kernel is called twice:
 *   pass 1: N elements            -> numberOfBlocks partials
 *   pass 2: numberOfBlocks partials -> 1 final value (single block launch)
 * It's the same kernel both times; pass 2 just launches it with
 * gridDim.x == 1 so "each block's result" IS the final result.
 */

// ---------------------------------------------------------------------
// map: a3[i] = f(a1[i], a2[i])
// ---------------------------------------------------------------------
__global__
void map_2kernel(float *a1, float *a2, float *a3, int size, float (*f)(float,float))
{
    const float* __restrict__ ra1 = a1;
    const float* __restrict__ ra2 = a2;
    float* __restrict__ ra3 = a3;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < size) {
        ra3[tid] = f(ra1[tid], ra2[tid]);
        tid = tid + (blockDim.x * gridDim.x);
    }
}

// ---------------------------------------------------------------------
// reduce: one block-level reduction. Each block writes its own result
// to out[blockIdx.x] - no atomics, no contention. Called twice from
// the host: once to reduce the N input elements down to
// numberOfBlocks partials, once more (as a single-block launch) to
// reduce those partials down to the final scalar.
// ---------------------------------------------------------------------
__global__
void reduce_kernel(float *a, float *out, float (*f)(float,float),
                    float identity, int n)
{
    const float* __restrict__ ra = a;
    float* __restrict__ rout = out;

    __shared__ float cache[256];
    int cacheIndex = threadIdx.x;
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    float temp = identity;
    while (tid < n) {
        temp = f(ra[tid], temp);
        tid = tid + (blockDim.x * gridDim.x);
    }
    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i)
            cache[cacheIndex] = f(cache[cacheIndex + i], cache[cacheIndex]);
        __syncthreads();
        i = i / 2;
    }

    if (cacheIndex == 0)
        rout[blockIdx.x] = cache[0];
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

    // pass 1: N elements -> numberOfBlocks partials
    // identity for `+` is 0.0f; pass -INFINITY for max, +INFINITY for min, etc.
    reduce_kernel<<<numberOfBlocks, threadsPerBlock>>>(dev_resp, d_partials, f2, 0.0f, N);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    // pass 2: numberOfBlocks partials -> 1 final value (single block)
    reduce_kernel<<<1, threadsPerBlock>>>(d_partials, d_final, f2, 0.0f, numberOfBlocks);
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

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("CUDA\t%d\t%3.1f\n", N, time);

    free(a);
    free(b);
    free(final);

    return 0;
}
