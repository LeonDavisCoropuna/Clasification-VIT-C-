#include <torch/torch.h>

class TransposeLayerImpl : public torch::nn::Module {
public:
    TransposeLayerImpl(int64_t dim1, int64_t dim2)
        : dim1_(dim1), dim2_(dim2) {}

    torch::Tensor forward(torch::Tensor x) {
        return x.transpose(dim1_, dim2_);
    }

private:
    int64_t dim1_;
    int64_t dim2_;
};

TORCH_MODULE(TransposeLayer);