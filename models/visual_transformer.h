#include <torch/torch.h>
#include <cmath>
#include "transformer.h"  // Contains your SelfAttention and Transformer implementations
#include "tokenizer.h"
#include "projector.h"
torch::Tensor positionalencoding1d(int64_t d_model, int64_t length) {
    if (d_model % 2 != 0) {
        throw std::invalid_argument("Cannot use sin/cos positional encoding with odd dim");
    }
    
    torch::Tensor pe = torch::zeros({length, d_model});
    torch::Tensor position = torch::arange(0, length).unsqueeze(1);
    torch::Tensor div_term = torch::exp(torch::arange(0, d_model, 2).to(torch::kFloat32) * 
                                      -(std::log(10000.0f) / d_model));
    
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                 torch::sin(position.to(torch::kFloat32) * div_term));
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                 torch::cos(position.to(torch::kFloat32) * div_term));
    
    return pe;
}

torch::Tensor positionalencoding2d(int64_t d_model, int64_t height, int64_t width) {
    if (d_model % 4 != 0) {
        throw std::invalid_argument("Cannot use sin/cos positional encoding with odd dimension");
    }
    
    torch::Tensor pe = torch::zeros({d_model, height, width});
    int64_t half_dim = d_model / 2;
    
    torch::Tensor div_term = torch::exp(torch::arange(0, half_dim, 2).to(torch::kFloat32) *
                           -(std::log(10000.0f) / half_dim));
    
    torch::Tensor pos_w = torch::arange(0., width).unsqueeze(1);
    torch::Tensor pos_h = torch::arange(0., height).unsqueeze(1);
    
    // Width-based encoding
    torch::Tensor sin_w = torch::sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat({1, height, 1});
    torch::Tensor cos_w = torch::cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat({1, height, 1});
    
    // Height-based encoding
    torch::Tensor sin_h = torch::sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat({1, 1, width});
    torch::Tensor cos_h = torch::cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat({1, 1, width});
    
    // Combine encodings
    pe.index_put_({torch::indexing::Slice(0, half_dim, 2), 
                  torch::indexing::Slice(), 
                  torch::indexing::Slice()}, sin_w);
    pe.index_put_({torch::indexing::Slice(1, half_dim, 2), 
                  torch::indexing::Slice(), 
                  torch::indexing::Slice()}, cos_w);
    pe.index_put_({torch::indexing::Slice(half_dim, torch::indexing::None, 2), 
                  torch::indexing::Slice(), 
                  torch::indexing::Slice()}, sin_h);
    pe.index_put_({torch::indexing::Slice(half_dim + 1, torch::indexing::None, 2), 
                  torch::indexing::Slice(), 
                  torch::indexing::Slice()}, cos_h);
    
    return pe;
}

class VisualTransformerImpl : public torch::nn::Module {
public:
    VisualTransformerImpl(
        int64_t in_channels,
        int64_t out_channels,
        int64_t token_channels,
        int64_t tokens,
        const std::string& tokenizer_type,
        int64_t attn_dim,
        int64_t transformer_enc_layers,
        int64_t transformer_heads,
        int64_t transformer_fc_dim,
        double transformer_dropout,
        bool is_projected = true
    ) : in_channels(in_channels),
        out_channels(out_channels),
        token_channels(token_channels),
        attn_dim(attn_dim),
        is_projected(is_projected),
        tokens(tokens),
        tokenizer_type(tokenizer_type) {
        
        if (tokenizer_type != "recurrent" && tokenizer_type != "filter") {
            throw std::invalid_argument("tokenizer type must be either recurrent or filter");
        }
        
        if (tokenizer_type == "recurrent") {
            tokenizer = register_module("tokenizer", 
                RecurrentTokenizer(in_channels, token_channels));
        } else {
            tokenizer = register_module("tokenizer", 
                FilterTokenizer(in_channels, token_channels, tokens));
        }
        
        transformer = register_module("transformer",
            Transformer(token_channels, attn_dim, transformer_dropout));
        
        if (is_projected) {
            projector = register_module("projector",
            Projector(in_channels, out_channels, token_channels));
        }
    }
    
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor t = torch::Tensor()) {
        // Tokenization
        torch::Tensor token_out;
        if (tokenizer_type == "filter") {
          token_out = tokenizer.forward<Tensor>(x);
        } else {
          token_out = tokenizer.forward<Tensor>(x, t);
        }
        
        // Transformer expects (L, N, C)
        token_out = token_out.permute({1, 0, 2});
        
        // Apply transformer (self-attention)
        auto t_out = transformer->forward(token_out);
        
        // Return to (N, L, C) shape
        t_out = t_out.permute({1, 0, 2});
        token_out = token_out.permute({1, 0, 2});
        
        torch::Tensor out;
        if (is_projected) {
            out = projector->forward(x, t_out);
            throw std::runtime_error("Projector not implemented yet");
        }
        
        return {out, token_out};
    }
    
private:
    int64_t in_channels;
    int64_t out_channels;
    int64_t token_channels;
    int64_t attn_dim;
    bool is_projected;
    int64_t tokens;
    std::string tokenizer_type;

    torch::nn::AnyModule tokenizer;  // Puede contener cualquiera de los dos tokenizers

    Projector projector{nullptr};
    
    Transformer transformer{nullptr};
};

TORCH_MODULE(VisualTransformer);