#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <unordered_map>
#include <random>
class ImageNetDataset
{
public:
  std::string train_data_dir = "/home/leon/Documentos/UNSA/TOPICOS IA/dataset/dataset/imagenet-mini/train";
  std::string valid_data_dir = "/home/leon/Documentos/UNSA/TOPICOS IA/dataset/dataset/imagenet-mini/val";

  std::vector<std::string> data;
  std::vector<int> labels;
  std::unordered_map<std::string, int> class_to_index;
  std::unordered_map<int, std::string> index_to_class;
  bool train = true;

  ImageNetDataset(bool is_train = true, int num_classes = 1000)
  {
    train = is_train;
    // Cargar imágenes y etiquetas
    if (train)
    {
      load_train_data(num_classes);
    }
    else
    {
      load_valid_data(num_classes);
    }
  }

  void load_train_data(int num_classes)
  {
    DIR *dir = opendir(train_data_dir.c_str());
    struct dirent *entry;

    int class_index = 0;
    while ((entry = readdir(dir)) != nullptr)
    {
      std::string class_dir = entry->d_name;
      if (class_dir == "." || class_dir == "..")
        continue; // Ignorar las carpetas "." y ".."

      // Asignamos un índice a la clase si aún no tiene uno
      if (class_to_index.find(class_dir) == class_to_index.end())
      {
        class_to_index[class_dir] = class_index;
        index_to_class[class_index] = class_dir;
        class_index++;
      }

      std::string class_path = train_data_dir + "/" + class_dir;
      load_data_from_dir(class_path, class_to_index[class_dir], num_classes);
    }
    closedir(dir);
  }

  void load_valid_data(int num_classes)
  {
    DIR *dir = opendir(valid_data_dir.c_str());
    struct dirent *entry;

    int class_index = 0;
    while ((entry = readdir(dir)) != nullptr)
    {
      std::string class_dir = entry->d_name;
      if (class_dir == "." || class_dir == "..")
        continue; // Ignorar las carpetas "." y ".."

      // Asignamos un índice a la clase si aún no tiene uno
      if (class_to_index.find(class_dir) == class_to_index.end())
      {
        class_to_index[class_dir] = class_index;
        index_to_class[class_index] = class_dir;
        class_index++;
      }

      std::string class_path = valid_data_dir + "/" + class_dir;
      load_data_from_dir(class_path, class_to_index[class_dir], num_classes);
    }
    closedir(dir);
  }

  void load_data_from_dir(const std::string &dir_path, int label, int num_classes)
  {
    DIR *dir = opendir(dir_path.c_str());
    struct dirent *entry;

    while ((entry = readdir(dir)) != nullptr)
    {
      if (entry->d_type == DT_REG)
      {
        data.push_back(dir_path + "/" + entry->d_name);
        labels.push_back(label);
      }
    }
    closedir(dir);
  }

  torch::Tensor load_image(const std::string &image_path)
  {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::resize(img, img, cv::Size(224, 224)); // Resize a 224x224 

    torch::Tensor image = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kUInt8).clone();
    image = image.permute({2, 0, 1}); // De HWC a CHW

    image = normalize_image(image);

    if (train)
    {
      image = random_horizontal_flip(image);
    }

    return image;
  }

  torch::Tensor normalize_image(torch::Tensor image)
  {
    image = image.to(torch::kFloat32).div(255); // Convert to float and scale to [0, 1]

    // Reshape mean and std to match image dimensions [3, 1, 1] for broadcasting
    auto mean = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
    auto std = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});

    return (image - mean) / std;
  }

  torch::Tensor random_horizontal_flip(torch::Tensor image, double p = 0.5)
  {
    static std::mt19937 gen(std::random_device{}());
    std::bernoulli_distribution dist(p);
    if (dist(gen))
    {
      image = image.flip({1}); // flip en ancho
    }
    return image;
  }

  std::pair<torch::Tensor, int> get_item(int index)
  {
    std::string image_path = data[index];
    torch::Tensor image = load_image(image_path);
    int label = labels[index];
    return {image, label}; // Devolver imagen y etiqueta
  }

  size_t size()
  {
    return data.size();
  }

  std::string get_class_name(int label)
  {
    return index_to_class[label]; // Devuelve el nombre de la clase basado en el índice
  }
};
