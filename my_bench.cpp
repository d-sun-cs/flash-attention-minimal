#include <torch/torch.h>
#include <iostream>

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor my_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

int main()
{
    const int batch_size = 16;
    const int n_head = 12;
    const int seq_len = 64;
    const int head_embd = 64;

    auto Q = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();
    auto K = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();
    auto V = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();

    auto O1 = forward(Q, K, V);
    auto O2 = my_forward(Q, K, V);

    // 1. 设置阈值
    float atol = 1e-3;
    float rtol = 1e-6;

    if (torch::allclose(O1, O2, atol, rtol))
    {
        std::cout << "O1 and O2 are equal!" << std::endl;
    }
    else
    {
        std::cout << "O1 and O2 are NOT equal!" << std::endl;
    }

    

    // 2. 找到不满足条件的掩码 (逻辑：|a - b| > atol + rtol * |b|)
    auto diff = torch::abs(O1 - O2);
    auto threshold = atol + rtol * torch::abs(O2);
    auto mask = diff > threshold;

    // 3. 统计不相等的点数
    auto num_mismatches = mask.sum().item<int64_t>();

    if (num_mismatches == 0)
    {
        std::cout << "All elements are equal within tolerance." << std::endl;
    }
    else
    {
        std::cout << "Found " << num_mismatches << " mismatched elements!" << std::endl;

        // 4. 获取不相等元素的索引 (N x 4 的张量，4 是因为你的维度是 [B, H, S, E])
        // nonzero() 会返回所有 True 元素的坐标
        auto indices = torch::nonzero(mask);

        // 5. 将数据移至 CPU 方便打印（极其重要，否则 GPU -> CPU 频繁同步会非常慢）
        auto O1_cpu = O1.to(torch::kCPU);
        auto O2_cpu = O2.to(torch::kCPU);
        auto indices_cpu = indices.to(torch::kCPU);

        // 6. 遍历并打印前 N 个不相等的值（防止屏幕被刷满）
        int max_print = 20;
        std::cout << std::left << std::setw(20) << "Index"
                  << std::setw(15) << "O1 Value"
                  << std::setw(15) << "O2 Value"
                  << std::setw(15) << "Diff" << std::endl;
        std::cout << std::string(65, '-') << std::endl;

        for (int i = 0; i < std::min((int)num_mismatches, max_print); ++i)
        {
            // 提取坐标
            auto idx = indices_cpu[i];
            int64_t b = idx[0].item<int64_t>();
            int64_t h = idx[1].item<int64_t>();
            int64_t s = idx[2].item<int64_t>();
            int64_t e = idx[3].item<int64_t>();

            // 提取具体数值
            float v1 = O1_cpu[b][h][s][e].item<float>();
            float v2 = O2_cpu[b][h][s][e].item<float>();

            std::cout << "[" << b << "," << h << "," << s << "," << e << "]\t"
                      << std::setw(15) << v1
                      << std::setw(15) << v2
                      << std::setw(15) << std::abs(v1 - v2) << std::endl;
        }

        if (num_mismatches > max_print)
        {
            std::cout << "... and " << (num_mismatches - max_print) << " more mismatches." << std::endl;
        }
    }

    return 0;
}