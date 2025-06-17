#include <torch/torch.h>
using namespace torch;

class FilterTokenizerImpl : nn::Module {
    int tokens;
    int in_channels;
    int token_channels;
    nn::Linear linear1{nullptr}, linear2{nullptr};
    Tensor cache1, cache2, token_cache;

    FilterTokenizerImpl(int in_channels, int token_channels, int tokens)
        : tokens(tokens), in_channels(in_channels), token_channels(token_channels) {

        linear1 = register_module("linear1", nn::Linear(in_channels, tokens));
        linear2 = register_module("linear2", nn::Linear(in_channels, token_channels));

        nn::init::xavier_normal_(linear1->weight);
        nn::init::xavier_normal_(linear2->weight);
    }

    Tensor forward(Tensor x) {
        Tensor a = linear1->forward(x);  // (N, HW, L)
        cache1 = a;
        a = a.softmax(1);                // softmax on HW
        cache2 = a;
        a = a.transpose(1, 2);           // (N, L, HW)
        a = torch::matmul(a, x);         // (N, L, C)
        a = linear2->forward(a);         // (N, L, D)
        token_cache = a;
        return a;
    }
};
TORCH_MODULE(FilterTokenizer);

class RecurrentTokenizerImpl : nn::Module {
    int token_channels;
    nn::Linear linear1{nullptr}, linear2{nullptr};
    Tensor cache1, cache2, token_cache;

    RecurrentTokenizerImpl(int in_channels, int token_channels)
        : token_channels(token_channels) {

        linear1 = register_module("linear1", nn::Linear(token_channels, token_channels));
        linear2 = register_module("linear2", nn::Linear(in_channels, token_channels));

        nn::init::xavier_normal_(linear1->weight);
        nn::init::xavier_normal_(linear2->weight);
    }

    Tensor forward(Tensor x, Tensor t) {
        Tensor a = linear1->forward(t);        // (N, L, D)
        Tensor b = linear2->forward(x);        // (N, HW, D)

        a = a.transpose(1, 2);                 // (N, D, L)
        a = torch::matmul(b, a);               // (N, HW, L)
        cache1 = a;
        a = a.softmax(1);                      // softmax over HW
        cache2 = a;
        a = a.transpose(1, 2);                 // (N, L, HW)
        b = torch::matmul(a, b);               // (N, L, D)

        return b;
    }
};
TORCH_MODULE(RecurrentTokenizer);
