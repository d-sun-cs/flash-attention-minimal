#include <torch/types.h>

__global__ void my_forward_kernel(const float *Q, const float *K, const float *V, const int N, const int d,
                                  const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                                  float *l, float *m, float *O)
{
    auto tid = threadIdx.x;
    auto bid = blockIdx.y * gridDim.x + blockIdx.x; // 第几张列表
    auto thread_num = blockDim.x * blockDim.y;

    // block start
    auto Q_start = Q + bid * N * d;
    auto K_start = K + bid * N * d;
    auto V_start = V + bid * N * d;
    auto O_start = O + bid * N * d;

    extern __shared__ float smem[];
    int offset = 0;
    int Qtile = Br * d;
    int KVtile = Bc * d;
    auto Qs = &smem[offset];
    offset += Qtile;
    auto Ks = &smem[offset];
    offset += KVtile;
    auto Vs = &smem[offset];
    offset += KVtile;
    auto Ss = &smem[offset];

    for (int i = 0; i < Tc; i++) {
        // K, V -> smem
        for (int j = 0; j < KVtile; j += thread_num) {
            // not coalesced and bank conflicts
            // Ks[tid * d + j] = K_start[i * Bc * d + tid * d + j];
            // Vs[tid * d + j] = V_start[i * Bc * d + tid * d + j];
            Ks[j + tid] = K_start[i * KVtile + j + tid];
            Vs[j + tid] = V_start[i * KVtile + j + tid];
        }
        __syncthreads();
        for (int j = 0; j < Tr; j++) {
            // Q -> smem
            for (int k = 0; k < Qtile; k += thread_num) {
                Qs[k + tid] = Q_start[j * Qtile + k + tid];
            }
            __syncthreads();
            // compute S
            auto row_m = -INFINITY;
            // one Q row per thread
            for (int l = 0; l < Bc; l++) {
                for (int m = 0; m < d; m++) {
                    // bank conflicts
                    Ss[tid * Bc + l] += Qs[tid * d + m] * Ks[l * d + m];
                }
                Ss[tid * Bc + l] *= softmax_scale;
                // one elem in one row
                if (Ss[tid * Bc + l] > row_m) {
                    row_m = Ss[tid * Bc + l];
                }
            }
        }

    }
}

torch::Tensor my_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    const int Bc = 32;
    const int Br = 32;

    // unknown vars
    const int B = Q.size(0);
    const int nh = Q.size(1);

    // why N, d are got by such way
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh); // batch_size x num_heads
    dim3 block_dim(Br);   // Bc ??? threads per block

    my_forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>());
    return O;
}