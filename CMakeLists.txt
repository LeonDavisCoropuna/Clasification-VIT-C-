cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_project)

# Configurar el estándar C++
set(CMAKE_CXX_STANDARD 17)

# Configuración de la ruta de LibTorch
set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")

# Ruta donde compilaste OpenCV
set(OpenCV_DIR "/home/leon/Documentos/UNSA/TOPICOS IA/opencv/build")  # Cambia a tu ruta real sin ~

find_package(OpenCV REQUIRED)

# Encontrar el paquete de LibTorch
find_package(Torch REQUIRED)

# Incluir directorios para tus archivos fuente y OpenCV
include_directories(${CMAKE_SOURCE_DIR}/data_loaders)
include_directories(${CMAKE_SOURCE_DIR}/models)

include_directories(${OpenCV_INCLUDE_DIRS})

# Agregar el archivo ejecutable
add_executable(main main.cpp data_loaders/image_net.h models/tokenizer.h models/projector.h models/custom_layers.h models/transformer.h models/visual_transformer.h models/resnet.h models/vt_resnet.h)

# Enlazar las bibliotecas de Torch y OpenCV al ejecutable
target_link_libraries(main "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

# Asegurarse de que el compilador use C++17 para Torch
set_property(TARGET main PROPERTY CXX_STANDARD 17)
