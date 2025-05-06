#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <unordered_map>

class ImageNetDataset
{
public:
  std::string train_data_dir = "/home/leon/Documentos/UNSA/TOPICOS IA/dataset/dataset/imagenet-mini/train";
  std::string valid_data_dir = "/home/leon/Documentos/UNSA/TOPICOS IA/dataset/dataset/imagenet-mini/val";

  std::vector<std::string> data;
  std::vector<int> labels;
  std::unordered_map<std::string, int> class_to_index;
  std::unordered_map<int, std::string> index_to_class;

  ImageNetDataset(bool train = true, int num_classes = 1000)
  {
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

    return torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kUInt8).clone();
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
