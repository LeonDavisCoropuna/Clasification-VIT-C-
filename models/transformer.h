#include <torch/torch.h>
#include <cmath>
#include <vector>

class SelfAttentionImpl : public torch::nn::Module {
public:
    SelfAttentionImpl(int64_t channels, int64_t attn_dim) 
        : channels(channels), attn_dim(attn_dim) {
        // Initialize query and key projections
        query_linear = register_module("query_linear", 
            torch::nn::Linear(channels, attn_dim));
        key_linear = register_module("key_linear", 
            torch::nn::Linear(channels, attn_dim));
        
        // Initialize weights with Kaiming normal (manual initialization)
        initialize_weights_kaiming(query_linear->weight);
        initialize_weights_kaiming(key_linear->weight);
        
        // Disable bias
        query_linear->bias = torch::Tensor();
        key_linear->bias = torch::Tensor();
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // x shape: [N, L, C]
        auto key = linear_forward(key_linear, x);      // [N, L, D]
        auto query = linear_forward(query_linear, x);  // [N, L, D]
        query = transpose(query, 1, 2);                // [N, D, L]
        
        auto T_out = matrix_multiply(key, query);      // [N, L, L]
        T_out = softmax(T_out, 1);                    // Softmax along dim 1
        
        T_out = matrix_multiply(T_out, x);             // [N, L, C]
        
        return T_out;
    }
    
private:
    int64_t channels;
    int64_t attn_dim;
    torch::nn::Linear query_linear{nullptr};
    torch::nn::Linear key_linear{nullptr};
    
    void initialize_weights_kaiming(torch::Tensor& weight) {
        // Manual Kaiming normal initialization
        auto fan_in = weight.size(1);
        auto std = sqrt(2.0 / fan_in);
        auto data = weight.data_ptr<float>();
        auto numel = weight.numel();
        
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = std * randn_like();
        }
    }
    
    float randn_like() {
        // Simple Box-Muller transform for normal distribution
        static bool has_spare = false;
        static float spare;
        
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        
        has_spare = true;
        static float u, v, s;
        do {
            u = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
            v = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
            s = u * u + v * v;
        } while (s >= 1.0f || s == 0.0f);
        
        s = sqrt(-2.0f * log(s) / s);
        spare = v * s;
        return u * s;
    }
    
    torch::Tensor linear_forward(torch::nn::Linear& linear, const torch::Tensor& input) {
        // Manual linear forward
        auto weight = linear->weight;  // [out_dim, in_dim]
        auto output = torch::zeros({input.size(0), input.size(1), weight.size(0)});
        
        auto input_a = input.accessor<float, 3>();
        auto weight_a = weight.accessor<float, 2>();
        auto output_a = output.accessor<float, 3>();
        
        for (int64_t n = 0; n < input.size(0); ++n) {
            for (int64_t l = 0; l < input.size(1); ++l) {
                for (int64_t o = 0; o < weight.size(0); ++o) {
                    float sum = 0.0f;
                    for (int64_t i = 0; i < weight.size(1); ++i) {
                        sum += input_a[n][l][i] * weight_a[o][i];
                    }
                    output_a[n][l][o] = sum;
                }
            }
        }
        
        return output;
    }
    
    torch::Tensor transpose(const torch::Tensor& input, int64_t dim1, int64_t dim2) {
        // Manual transpose
        auto sizes = input.sizes().vec();
        std::swap(sizes[dim1], sizes[dim2]);
        auto output = torch::zeros(sizes);
        
        auto input_a = input.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        
        for (int64_t n = 0; n < input.size(0); ++n) {
            for (int64_t i = 0; i < input.size(1); ++i) {
                for (int64_t j = 0; j < input.size(2); ++j) {
                    if (dim1 == 1 && dim2 == 2) {
                        output_a[n][j][i] = input_a[n][i][j];
                    } else if (dim1 == 0 && dim2 == 1) {
                        output_a[i][n][j] = input_a[n][i][j];
                    } // ... other cases
                }
            }
        }
        
        return output;
    }
    
    torch::Tensor matrix_multiply(const torch::Tensor& a, const torch::Tensor& b) {
        // Manual matrix multiplication
        auto output = torch::zeros({a.size(0), a.size(1), b.size(2)});
        
        auto a_a = a.accessor<float, 3>();
        auto b_a = b.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        
        for (int64_t n = 0; n < a.size(0); ++n) {
            for (int64_t i = 0; i < a.size(1); ++i) {
                for (int64_t j = 0; j < b.size(2); ++j) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < a.size(2); ++k) {
                        sum += a_a[n][i][k] * b_a[n][k][j];
                    }
                    output_a[n][i][j] = sum;
                }
            }
        }
        
        return output;
    }
    
    torch::Tensor softmax(const torch::Tensor& input, int64_t dim) {
        // Manual softmax
        auto output = torch::zeros_like(input);
        
        auto input_a = input.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        
        for (int64_t n = 0; n < input.size(0); ++n) {
            for (int64_t i = 0; i < input.size(1); ++i) {
                // Find max for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < input.size(dim); ++j) {
                    float val = (dim == 1) ? input_a[n][j][i] : input_a[n][i][j];
                    if (val > max_val) max_val = val;
                }
                
                // Compute exponentials and sum
                float sum = 0.0f;
                for (int64_t j = 0; j < input.size(dim); ++j) {
                    float val = (dim == 1) ? input_a[n][j][i] : input_a[n][i][j];
                    float exp_val = exp(val - max_val);
                    sum += exp_val;
                    if (dim == 1) output_a[n][j][i] = exp_val;
                    else output_a[n][i][j] = exp_val;
                }
                
                // Normalize
                for (int64_t j = 0; j < input.size(dim); ++j) {
                    if (dim == 1) output_a[n][j][i] /= sum;
                    else output_a[n][i][j] /= sum;
                }
            }
        }
        
        return output;
    }
};

TORCH_MODULE(SelfAttention);

class TransformerImpl : public torch::nn::Module {
public:
    TransformerImpl(int64_t token_channels, int64_t attn_dim, double dropout)
        : token_channels(token_channels), attn_dim(attn_dim) {
        // Initialize attention mechanism
        attention = register_module("attention", 
            SelfAttention(token_channels, attn_dim));
        
        // Initialize feed-forward layers
        linear1 = register_module("linear1", 
            torch::nn::Linear(token_channels, token_channels));
        linear2 = register_module("linear2", 
            torch::nn::Linear(token_channels, token_channels));
        
        // Initialize weights with Kaiming normal (manual)
        initialize_weights_kaiming(linear1->weight);
        initialize_weights_kaiming(linear2->weight);
        
        // Initialize layer norms
        layer_norm1 = register_module("layer_norm1", 
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({token_channels})));
        layer_norm2 = register_module("layer_norm2", 
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({token_channels})));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // x shape: [N, L, D]
        auto a = x + attention->forward(x);     // [N, L, D]
        a = manual_layer_norm(layer_norm1, a);  // [N, L, D]
        
        auto b = linear_forward(linear1, a);   // [N, L, D]
        b = relu(b);
        b = linear_forward(linear2, b);        // [N, L, D]
        a = a + b;                             // [N, L, D]
        a = manual_layer_norm(layer_norm2, a);  // [N, L, D]
        
        return a;
    }
    
private:
    int64_t token_channels;
    int64_t attn_dim;
    SelfAttention attention{nullptr};
    torch::nn::Linear linear1{nullptr};
    torch::nn::Linear linear2{nullptr};
    torch::nn::LayerNorm layer_norm1{nullptr};
    torch::nn::LayerNorm layer_norm2{nullptr};
    
    void initialize_weights_kaiming(torch::Tensor& weight) {
        // Same as in SelfAttention
        auto fan_in = weight.size(1);
        auto std = sqrt(2.0 / fan_in);
        auto data = weight.data_ptr<float>();
        auto numel = weight.numel();
        
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = std * randn_like();
        }
    }
    
    float randn_like() {
        // Same as in SelfAttention
        static bool has_spare = false;
        static float spare;
        
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        
        has_spare = true;
        static float u, v, s;
        do {
            u = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
            v = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
            s = u * u + v * v;
        } while (s >= 1.0f || s == 0.0f);
        
        s = sqrt(-2.0f * log(s) / s);
        spare = v * s;
        return u * s;
    }
    
    torch::Tensor linear_forward(torch::nn::Linear& linear, const torch::Tensor& input) {
        // Same as in SelfAttention
        auto weight = linear->weight;
        auto output = torch::zeros({input.size(0), input.size(1), weight.size(0)});
        
        auto input_a = input.accessor<float, 3>();
        auto weight_a = weight.accessor<float, 2>();
        auto output_a = output.accessor<float, 3>();
        
        for (int64_t n = 0; n < input.size(0); ++n) {
            for (int64_t l = 0; l < input.size(1); ++l) {
                for (int64_t o = 0; o < weight.size(0); ++o) {
                    float sum = 0.0f;
                    for (int64_t i = 0; i < weight.size(1); ++i) {
                        sum += input_a[n][l][i] * weight_a[o][i];
                    }
                    output_a[n][l][o] = sum;
                }
            }
        }
        
        return output;
    }
    
    torch::Tensor relu(const torch::Tensor& input) {
        // Manual ReLU
        auto output = torch::zeros_like(input);
        auto input_a = input.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        
        for (int64_t n = 0; n < input.size(0); ++n) {
            for (int64_t l = 0; l < input.size(1); ++l) {
                for (int64_t c = 0; c < input.size(2); ++c) {
                    output_a[n][l][c] = std::max(0.0f, input_a[n][l][c]);
                }
            }
        }
        
        return output;
    }
    
    torch::Tensor manual_layer_norm(torch::nn::LayerNorm& ln, const torch::Tensor& input) {
        // Manual layer norm (simplified)
        auto output = torch::zeros_like(input);
        auto input_a = input.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        
        for (int64_t n = 0; n < input.size(0); ++n) {
            for (int64_t l = 0; l < input.size(1); ++l) {
                // Compute mean
                float mean = 0.0f;
                for (int64_t c = 0; c < input.size(2); ++c) {
                    mean += input_a[n][l][c];
                }
                mean /= input.size(2);
                
                // Compute variance
                float var = 0.0f;
                for (int64_t c = 0; c < input.size(2); ++c) {
                    var += (input_a[n][l][c] - mean) * (input_a[n][l][c] - mean);
                }
                var /= input.size(2);
                float std = sqrt(var + 1e-5f);
                
                // Normalize
                for (int64_t c = 0; c < input.size(2); ++c) {
                    output_a[n][l][c] = (input_a[n][l][c] - mean) / std;
                }
            }
        }
        
        return output;
    }
};

TORCH_MODULE(Transformer);