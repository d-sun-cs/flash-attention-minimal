#include <torch/torch.h>
#include <iostream>

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

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

    return 0;
}