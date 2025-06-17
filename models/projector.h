#include <torch/torch.h>

class ProjectorImpl : public torch::nn::Module {
public:
    int64_t in_channels, out_channels, token_channels;
    torch::Tensor cache;

    torch::nn::Linear linear1{nullptr}, linear2{nullptr}, linear3{nullptr};
    torch::nn::BatchNorm1d norm{nullptr};
    torch::nn::Sequential downsample{nullptr};

    ProjectorImpl(int64_t in_channels, int64_t out_channels, int64_t token_channels)
        : in_channels(in_channels), out_channels(out_channels), token_channels(token_channels) {
        
        linear1 = register_module("linear1", torch::nn::Linear(torch::nn::LinearOptions(in_channels, token_channels).bias(false)));
        linear2 = register_module("linear2", torch::nn::Linear(torch::nn::LinearOptions(token_channels, token_channels).bias(false)));
        linear3 = register_module("linear3", torch::nn::Linear(token_channels, out_channels));

        torch::nn::init::xavier_normal_(linear1->weight);
        torch::nn::init::xavier_normal_(linear2->weight);
        torch::nn::init::xavier_normal_(linear3->weight);

        norm = register_module("norm", torch::nn::BatchNorm1d(out_channels));

        if (in_channels != out_channels) {
            downsample = register_module("downsample", torch::nn::Sequential(
                torch::nn::Linear(in_channels, out_channels)
            ));
        }
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor t) {
        auto x_q = linear1->forward(x); // (N, HW, token_channels)
        auto t_q = linear2->forward(t); // (N, L, token_channels)

        t_q = t_q.transpose(1, 2); // (N, token_channels, L)
        auto a = torch::matmul(x_q, t_q); // (N, HW, L)
        a = torch::softmax(a, 2);
        cache = a.clone();

        auto t_v = linear3->forward(t); // (N, L, out_channels)
        a = torch::matmul(a, t_v); // (N, HW, out_channels)

        if (downsample) {
            x = downsample->forward(x); // (N, HW, out_channels)
        }

        x = x + a; // residual connection

        x = x.transpose(1, 2); // (N, out_channels, HW)
        x = norm->forward(x);
        x = x.transpose(1, 2); // (N, HW, out_channels)
        x = torch::relu(x);

        return x;
    }
};

TORCH_MODULE(Projector);