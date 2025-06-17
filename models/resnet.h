#include <torch/torch.h>
#include <vector>
#include <stdexcept>

// 3x3 convolution with padding
torch::nn::Conv2d conv3x3(int in_planes, int out_planes, int stride = 1,
                          int groups = 1, int dilation = 1)
{
  torch::nn::Conv2dOptions options(in_planes, out_planes, 3);
  options.stride(stride);
  options.padding(dilation);
  options.groups(groups);
  options.dilation(dilation);
  options.bias(false);
  return torch::nn::Conv2d(options);
}

// 1x1 convolution
torch::nn::Conv2d conv1x1(int in_planes, int out_planes, int stride = 1)
{
  torch::nn::Conv2dOptions options(in_planes, out_planes, 1);
  options.stride(stride);
  options.bias(false);
  return torch::nn::Conv2d(options);
}

class BasicBlockImpl : public torch::nn::Module
{
public:
  static const int expansion = 1;

  BasicBlockImpl(int inplanes, int planes, int stride = 1,
                 torch::nn::Sequential downsample = nullptr,
                 int groups = 1, int base_width = 64,
                 int dilation = 1,
                 torch::nn::BatchNorm2d norm_layer = nullptr)
      : conv1(conv3x3(inplanes, planes, stride)),
        bn1(planes),
        conv2(conv3x3(planes, planes)),
        bn2(planes),
        downsample(downsample),
        stride(stride)
  {

    if (!norm_layer)
    {
      bn1 = torch::nn::BatchNorm2d(planes);
      bn2 = torch::nn::BatchNorm2d(planes);
    }
    else
    {
      bn1 = norm_layer;
      bn2 = norm_layer;
    }

    if (groups != 1 || base_width != 64)
    {
      throw std::invalid_argument("BasicBlock only supports groups=1 and base_width=64");
    }
    if (dilation > 1)
    {
      throw std::runtime_error("Dilation > 1 not supported in BasicBlock");
    }

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    if (!downsample.is_empty())
    {
      register_module("downsample", downsample);
    }
  }

  torch::Tensor forward(torch::Tensor x)
  {
    torch::Tensor identity = x;

    torch::Tensor out = conv1->forward(x);
    out = bn1->forward(out);
    out = torch::relu(out);

    out = conv2->forward(out);
    out = bn2->forward(out);

    if (!downsample.is_empty())
    {
      identity = downsample->forward(x);
    }

    out += identity;
    out = torch::relu(out);

    return out;
  }

public:
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::BatchNorm2d bn2{nullptr};
  torch::nn::Sequential downsample{nullptr};
  int stride;
};

TORCH_MODULE(BasicBlock);

class BottleneckImpl : public torch::nn::Module
{
public:
  static const int expansion = 4;

  BottleneckImpl(int inplanes, int planes, int stride = 1,
                 torch::nn::Sequential downsample = nullptr,
                 int groups = 1, int base_width = 64,
                 int dilation = 1,
                 torch::nn::BatchNorm2d norm_layer = nullptr)
      : conv1(conv1x1(inplanes, width(planes, base_width, groups))),
        bn1(width(planes, base_width, groups)),
        conv2(conv3x3(width(planes, base_width, groups),
                      width(planes, base_width, groups),
                      stride, groups, dilation)),
        bn2(width(planes, base_width, groups)),
        conv3(conv1x1(width(planes, base_width, groups), planes * expansion)),
        bn3(planes * expansion),
        downsample(downsample),
        stride(stride)
  {

    if (norm_layer.is_empty())
    {
      bn1 = torch::nn::BatchNorm2d(width(planes, base_width, groups));
      bn2 = torch::nn::BatchNorm2d(width(planes, base_width, groups));
      bn3 = torch::nn::BatchNorm2d(planes * expansion);
    }
    else
    {
      bn1 = norm_layer;
      bn2 = norm_layer;
      bn3 = norm_layer;
    }

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    if (!downsample.is_empty())
    {
      register_module("downsample", downsample);
    }
  }

  torch::Tensor forward(torch::Tensor x)
  {
    torch::Tensor identity = x;

    torch::Tensor out = conv1->forward(x);
    out = bn1->forward(out);
    out = torch::relu(out);

    out = conv2->forward(out);
    out = bn2->forward(out);
    out = torch::relu(out);

    out = conv3->forward(out);
    out = bn3->forward(out);

    if (!downsample.is_empty())
    {
      identity = downsample->forward(x);
    }

    out += identity;
    out = torch::relu(out);

    return out;
  }

public:
  int width(int planes, int base_width, int groups)
  {
    return int(planes * (base_width / 64.0)) * groups;
  }

  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::BatchNorm2d bn2{nullptr};
  torch::nn::Conv2d conv3{nullptr};
  torch::nn::BatchNorm2d bn3{nullptr};
  torch::nn::Sequential downsample{nullptr};
  int stride;
};

TORCH_MODULE(Bottleneck);

class ResNetImpl : public torch::nn::Module
{
public:
  ResNetImpl(
      BasicBlock block,
      std::vector<int> layers,
      int num_classes = 1000,
      bool backbone = true,
      bool custom_class_num = false,
      bool zero_init_residual = false,
      int groups = 1,
      int width_per_group = 64,
      std::vector<bool> replace_stride_with_dilation = {},
      torch::nn::BatchNorm2d norm_layer = nullptr) : block(block),
                                                     num_classes(num_classes),
                                                     backbone(backbone),
                                                     custom_class_num(custom_class_num),
                                                     groups(groups),
                                                     base_width(width_per_group)
  {

    if (norm_layer.is_empty())
    {
      this->norm_layer = torch::nn::BatchNorm2d(64);
    }
    else
    {
      this->norm_layer = norm_layer;
    }

    if (replace_stride_with_dilation.empty())
    {
      replace_stride_with_dilation = {false, false, false};
    }
    if (replace_stride_with_dilation.size() != 3)
    {
      throw std::invalid_argument("replace_stride_with_dilation should be empty or a 3-element vector");
    }

    inplanes = 64;
    dilation = 1;

    if (custom_class_num)
    {
      num_classes = 1000;
    }

    // Initial layers
    conv1 = register_module("conv1", torch::nn::Conv2d(
                                         torch::nn::Conv2dOptions(3, inplanes, 7)
                                             .stride(2)
                                             .padding(3)
                                             .bias(false)));

    bn1 = register_module("bn1", torch::nn::BatchNorm2d(inplanes));
    relu = register_module("relu", torch::nn::Functional(torch::relu));
    maxpool = register_module("maxpool", torch::nn::MaxPool2d(
                                             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

    // Layer blocks
    layer1 = register_module("layer1",
                             _make_layer(block, 64, layers[0]));
    layer2 = register_module("layer2",
                             _make_layer(block, 128, layers[1], 2, replace_stride_with_dilation[0]));
    layer3 = register_module("layer3",
                             _make_layer(block, 256, layers[2], 2, replace_stride_with_dilation[1]));
    layer4 = register_module("layer4",
                             _make_layer(block, 512, layers[3], 2, replace_stride_with_dilation[2]));

    // Final layers
    avgpool = register_module("avgpool",
                              torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    fc = register_module("fc",
                         torch::nn::Linear(512 * block->expansion, num_classes));

    // Initialize weights
    for (auto &module : modules())
    {
      if (auto M = dynamic_cast<torch::nn::Conv2dImpl *>(module.get()))
      {
        torch::nn::init::kaiming_normal_(M->weight, 0, torch::kFanOut, torch::kReLU);
      }
      else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl *>(module.get()))
      {
        torch::nn::init::constant_(M->weight, 1);
        torch::nn::init::constant_(M->bias, 0);
      }
    }

    // Zero-initialize the last BN in each residual branch
    if (zero_init_residual)
    {
      for (auto &module : modules())
      {
        if (auto M = dynamic_cast<BottleneckImpl *>(module.get()))
        {
          torch::nn::init::constant_(M->bn3->weight, 0);
        }
        else if (auto M = dynamic_cast<BasicBlockImpl *>(module.get()))
        {
          torch::nn::init::constant_(M->bn2->weight, 0);
        }
      }
    }
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = relu->forward(x);
    x = maxpool->forward(x);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);

    if (backbone)
    {
      return x;
    }

    x = layer4->forward(x);
    x = avgpool->forward(x);
    x = torch::flatten(x, 1);

    if (custom_class_num)
    {
      return x;
    }

    x = fc->forward(x);

    return x;
  }

private:
  torch::nn::Sequential _make_layer(
      BasicBlock block,
      int planes,
      int blocks,
      int stride = 1,
      bool dilate = false)
  {
    torch::nn::Sequential downsample = nullptr;
    int previous_dilation = dilation;

    if (dilate)
    {
      dilation *= stride;
      stride = 1;
    }

    if (stride != 1 || inplanes != planes * block->expansion)
    {
      downsample = torch::nn::Sequential(
          conv1x1(inplanes, planes * block->expansion, stride),
          norm_layer(planes * block->expansion));
    }

    torch::nn::Sequential layers;
    // Usar BasicBlock directamente como módulo
    layers->push_back(BasicBlock(inplanes, planes, stride, downsample, groups,
                                 base_width, previous_dilation, norm_layer));

    inplanes = planes * block->expansion;
    for (int i = 1; i < blocks; ++i)
    {
      layers->push_back(BasicBlock(inplanes, planes, groups, base_width, dilation,
                                   norm_layer));
    }

    return layers;
  }

  // Member variables
  BasicBlock block;
  int num_classes;
  bool backbone;
  bool custom_class_num;
  int groups;
  int base_width;
  torch::nn::BatchNorm2d norm_layer;
  int inplanes;
  int dilation;

  // Layers
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Functional relu{nullptr};
  torch::nn::MaxPool2d maxpool{nullptr};
  torch::nn::Sequential layer1{nullptr};
  torch::nn::Sequential layer2{nullptr};
  torch::nn::Sequential layer3{nullptr};
  torch::nn::Sequential layer4{nullptr};
  torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
  torch::nn::Linear fc{nullptr};
};

TORCH_MODULE(ResNet);