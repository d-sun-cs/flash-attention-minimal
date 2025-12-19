#include <torch/types.h>

__global__ void my_forward_kernel(const float *Q, const float *K, const float *V, const int N, const int d,
                                  const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                                  float *l, float *m, float *O)
{
    auto tid = threadIdx.x;
    auto bid = blockIdx.y * gridDim.x + blockIdx.x; // 第几张列表
    // auto bid = blockIdx.x * gridDim.y + blockIdx.y;
    auto thread_num = blockDim.x * blockDim.y;

    // block start
    auto Q_start = Q + bid * N * d;
    auto K_start = K + bid * N * d;
    auto V_start = V + bid * N * d;
    auto O_start = O + bid * N * d;
    auto m_start = m + bid * N;
    auto l_start = l + bid * N;

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

    for (auto i = 0; i < Tc; i++)
    {
        // K, V -> smem
        for (auto j = 0; j < KVtile; j += thread_num)
        {
            Ks[j + tid] = K_start[i * KVtile + j + tid];
            Vs[j + tid] = V_start[i * KVtile + j + tid];
        }
        // for (auto j = 0; j < d; j++)
        // {
        //     // not coalesced and bank conflicts
        //     Ks[tid * d + j] = K_start[i * KVtile + tid * d + j];
        //     Vs[tid * d + j] = V_start[i * KVtile + tid * d + j];
        // }
        __syncthreads();

        for (auto j = 0; j < Tr; j++)
        {
            // Q -> smem (acctually it's not shared)
            for (auto k = 0; k < Qtile; k += thread_num)
            {
                Qs[k + tid] = Q_start[j * Qtile + k + tid];
            }
            // for (auto k = 0; k < d; k++)
            // {
            //     Qs[tid * d + k] = Q_start[j * Qtile + tid * d + k];
            // }
            __syncthreads();

            // compute S
            // one Q row per thread
            auto row_m = -INFINITY; // thread priavte row max
            for (auto x = 0; x < Bc; x++)
            {
                float sum = 0; // zero initialized
                for (auto y = 0; y < d; y++)
                {
                    // bank conflicts
                    // Ss[tid * Bc + x] += Qs[tid * d + y] * Ks[x * d + y];
                    sum += Qs[tid * d + y] * Ks[x * d + y];
                }
                // Ss[tid * Bc + x] *= softmax_scale;
                sum *= softmax_scale;
                Ss[tid * Bc + x] = sum;
                // one elem in one row
                row_m = max(row_m, sum);
            }

            float row_l = 0;
            for (auto k = 0; k < Bc; k++)
            {
                // caclulate P (stored in Ss)
                Ss[tid * Bc + k] = __expf(Ss[tid * Bc + k] - row_m);
                row_l += Ss[tid * Bc + k];
            }

            // update row max and sum
            auto row_m_prev = m_start[j * Br + tid];
            auto row_l_prev = l_start[j * Br + tid];
            auto row_m_new = max(row_m, row_m_prev);
            auto row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev + __expf(row_m - row_m_new) * row_l;

            // White O, l, m to HBM
            for (auto x = 0; x < d; x++)
            {
                float PV = 0;
                for (auto y = 0; y < Bc; y++)
                {
                    PV += Ss[tid * Bc + y] * Vs[y * d + x];
                }
                O_start[(j * Br + tid) * d + x] = (1 / row_l_new) * (row_l_prev * __expf(row_m_prev - row_m_new) * O_start[(j * Br + tid) * d + x] + __expf(row_m - row_m_new) * PV);
            }
            m_start[j * Br + tid] = row_m_new;
            l_start[j * Br + tid] = row_l_new;
        }
        __syncthreads();
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
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh); // batch_size x num_heads
    dim3 block_dim(Br);   // Bc ??? threads per block

    my_forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>());
    return O;
}