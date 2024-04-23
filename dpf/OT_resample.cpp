#include <torch/extension.h>
#include <vector>



#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> sinkhorn_forward(
    torch::Tensor log_a,
    torch::Tensor log_b,
    torch::Tensor cost,
    float epsilon,
    float threshold,
    int max_iter,
    torch::Tensor diam,
    float rate
){
    int i = 0;
    auto fi = torch::zeroslike(log_a)
    auto gi = torch::zeroslike(log_b)
    for (int i = 0; i<5; i++) {
        
    }
}


std::vector<torch::Tensor> OT_forward(
    torch::Tensor xt,
    torch::Tensor logwt,
    float epsilon,
    float threshold,
    int max_iter,
    float rate
){
    const auto N = xt.size(1);

}
