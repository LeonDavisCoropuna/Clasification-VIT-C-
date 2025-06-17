#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "data_loaders/image_net.h"

void show_image(torch::Tensor &img_tensor)
{
    // Convert tensor to CPU and uint8
    torch::Tensor cpu_tensor = img_tensor.to(torch::kCPU).to(torch::kU8);
    
    // Permute from CHW to HWC format for OpenCV
    cpu_tensor = cpu_tensor.permute({1, 2, 0});
    
    // Get the dimensions of the tensor
    int height = cpu_tensor.size(0);
    int width = cpu_tensor.size(1);
    int channels = cpu_tensor.size(2);

    // Create a cv::Mat object with the same data type and dimensions as the tensor
    cv::Mat cv_img(height, width, CV_8UC(channels));

    // Copy the data from the tensor to the cv::Mat object
    std::memcpy(cv_img.data, cpu_tensor.data_ptr(), sizeof(torch::kU8) * cpu_tensor.numel());

    // Convert from BGR to RGB if needed (OpenCV uses BGR by default)
    cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);

    // Show the image
    cv::imshow("Image", cv_img);
    cv::waitKey(0);
}

int main()
{
  ImageNetDataset dataset(true, 1000);
  // Verificar si hay datos en el dataset
  std::cout << "Dataset size: " << dataset.size() << std::endl;

  // Verificar el índice 0
  if (dataset.size() > 0)
  {
    auto [sample, label] = dataset.get_item(1000);

    std::cout << "Sample tensor: " << sample.sizes() << std::endl;
    std::cout << "label: " << label;
    // Mostrar la imagen
    show_image(sample);
  }
  else
  {
    std::cerr << "No data available in the dataset." << std::endl;
  }

  return 0;
}
