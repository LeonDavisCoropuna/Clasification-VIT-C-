#include <torch/torch.h>
#include <vector>
#include <string>
#include "resnet.h"             // Assuming you have ResNet and BasicBlock implemented
#include "visual_transformer.h" // Assuming you have VisualTransformer implemented

class VTResNetImpl : public torch::nn::Module
{
public:
  VTResNetImpl(
      torch::nn::ModuleHolder<ResNet> resnet_layer,
      int vt_layers_num,
      int tokens,
      int token_channels,
      int input_dim,
      int vt_channels,
      int transformer_enc_layers,
      int transformer_heads,
      int transformer_fc_dim = 1024,
      double transformer_dropout = 0.5,
      int image_channels = 3,
      int num_classes = 1000) : in_channels(vt_channels / 2),
                                tokens(tokens),
                                vt_channels(vt_channels),
                                vt_layers_num(vt_layers_num),
                                vt_layer_res(input_dim / 16)
  {

    // Register modules
    resnet = register_module("resnet", resnet_layer);
    bn = register_module("bn", torch::nn::BatchNorm2d(in_channels));

    // First VT layer
    vt_layers = register_module("vt_layers", torch::nn::ModuleList());
    vt_layers->push_back(
        VisualTransformer(
            in_channels,
            vt_channels,
            token_channels,
            tokens,
            "filter",       // tokenizer_type
            token_channels, // attn_dim
            transformer_enc_layers,
            transformer_heads,
            transformer_fc_dim,
            transformer_dropout,
            true // is_projected
            ));

    // Additional VT layers
    for (int i = 1; i < vt_layers_num; ++i)
    {
      vt_layers->push_back(
          VisualTransformer(
              vt_channels,
              vt_channels,
              token_channels,
              tokens,
              "recurrent",    // tokenizer_type
              token_channels, // attn_dim
              transformer_enc_layers,
              transformer_heads,
              transformer_fc_dim,
              transformer_dropout,
              true // is_projected
              ));
    }

    avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    fc = register_module("fc", torch::nn::Linear(vt_channels, num_classes));

    // Initialize weights
    for (auto &module : modules())
    {
      if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl *>(module.get()))
      {
        torch::nn::init::constant_(M->weight, 1);
        torch::nn::init::constant_(M->bias, 0);
      }
    }
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = resnet->forward(x);
    x = bn->forward(x);

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Flatten pixels
    x = torch::flatten(x, 2);
    x = x.permute({0, 2, 1});

    auto [out, t] = vt_layers[0]->as<VisualTransformer>()->forward(x);

    for (int i = 1; i < vt_layers_num; ++i)
    {
      std::tie(out, t) = vt_layers[i]->as<VisualTransformer>()->forward(out, t);
    }

    out = out.permute({0, 2, 1});
    out = out.reshape({N, vt_channels, vt_layer_res, vt_layer_res});

    out = avgpool->forward(out);
    out = torch::flatten(out, 1);
    out = fc->forward(out);

    return out;
  }

private:
  int in_channels;
  int tokens;
  int vt_channels;
  int vt_layers_num;
  int vt_layer_res;

  torch::nn::ModuleHolder<ResNet> resnet{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  torch::nn::ModuleList vt_layers{nullptr};
  torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
  torch::nn::Linear fc{nullptr};
};

TORCH_MODULE(VTResNet);

VTResNet create_model(
    const std::string &arch,
    const std::vector<int> &layers,
    const std::string &freeze,
    bool progress,
    int vt_layers_num,
    int tokens,
    int token_channels,
    int input_dim,
    int vt_channels,
    int transformer_enc_layers,
    int transformer_heads,
    int transformer_fc_dim = 1024,
    double transformer_dropout = 0.5,
    int image_channels = 3,
    int num_classes = 1000)
{
  if (freeze != "no_freeze" && freeze != "partial_freeze" && freeze != "full_freeze")
  {
    throw std::invalid_argument("Freeze value undefined");
  }

  // Crear ResNet con los parámetros correctos
  BasicBlock block;
  auto resnet = ResNet(
      block,
      layers,
      1000,   // num_classes
      true,   // backbone
      false,  // custom_class_num
      false,  // zero_init_residual
      1,      // groups
      64,     // width_per_group
      {},     // replace_stride_with_dilation
      nullptr // norm_layer
  );

  // Initialize weights if not pretrained
  for (auto &module : resnet->modules())
  {
    if (auto M = dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
    {
      torch::nn::init::kaiming_normal_(M->weight, 0, torch::kFanOut, torch::kReLU);
    }
  }

  return VTResNet(
      resnet,
      vt_layers_num,
      tokens,
      token_channels,
      input_dim,
      vt_channels,
      transformer_enc_layers,
      transformer_heads,
      transformer_fc_dim,
      transformer_dropout,
      image_channels,
      num_classes);
}

VTResNet vt_resnet18(
    bool pretrained = false,
    const std::string &freeze = "no_freeze",
    bool progress = true,
    int tokens = 16,
    int token_channels = 256,
    int input_dim = 224,
    int vt_channels = 512,
    int transformer_enc_layers = 4,
    int transformer_heads = 8,
    int transformer_fc_dim = 1024,
    double transformer_dropout = 0.5,
    int image_channels = 3,
    int num_classes = 1000)
{
  return create_model(
      "resnet18",
      {2, 2, 2, 2},
      freeze,
      progress,
      /* vt_layers_num */ 2, // layers[-1] from Python becomes 2 for resnet18
      tokens,
      token_channels,
      input_dim,
      vt_channels,
      transformer_enc_layers,
      transformer_heads,
      transformer_fc_dim,
      transformer_dropout,
      image_channels,
      num_classes);
}