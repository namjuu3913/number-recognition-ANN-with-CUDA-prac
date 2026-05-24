#include <complex>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>
#include <fstream>
#include <algorithm>
#include "ANN.h"

/*
 * Changes Big-Endian into Little-Endian
 * */
uint32_t swap_endian(uint32_t val)
{
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >>16);
}

/*
 * Image data loader
 * (vibe coded)
 * */
std::vector<float> read_mnist_images(std::string path, int& count)
{
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) return {};

  uint32_t magic, num_images, rows, cols;
  file.read((char*)&magic, 4);
  file.read((char*)&num_images, 4);
  file.read((char*)&rows, 4);
  file.read((char*)&cols, 4);

  num_images = swap_endian(num_images);
  count = num_images;
  rows = swap_endian(rows);
  cols = swap_endian(cols);

  std::vector<float> images(num_images * 784);
  for (int i = 0; i < num_images * 784; ++i) 
  {
      unsigned char pixel = 0;
      file.read((char*)&pixel, 1);
      images[i] = pixel / 255.0f; // Noramlize between 0~1
  }
  return images;
}

/*
 * Label loader
 * */
std::vector<int> read_mnist_labels(std::string path, int& count)
{
  std::ifstream file(path, std::ios::binary);

  if(!file.is_open())
    return {};

  uint32_t magic, number_labels;
  file.read((char*)&magic, 4);
  file.read((char*)&number_labels, 4);

  number_labels =  swap_endian(number_labels);
  count = number_labels;

  std::vector<int> labels_re(number_labels);
  for(int i = 0; i < number_labels; i++)
  {
    unsigned char label_loop = 0;
    file.read((char*)&labels_re, 1);
    labels_re[i] = (int)label_loop;
  }

  return labels_re;
}

int main() 
{
  int image_count = 0;
  int label_count = 0;
  std::cout << "Laoding MNIST data........." << std::endl;

  // 1. Data load (training)
  std::vector<float> train_images = read_mnist_images("dataset/train/train-images.idx3-ubyte", image_count);
  std::vector<int> train_labels = read_mnist_labels("dataset/train/train-labels.idx1-ubyte", label_count);

  if(train_images.empty()||train_labels.empty())
  {
    std::cerr << "Cannot find the file. Check the Path again!" << std::endl;
    return -1;
  }

  // 2. initialize ANN (input 784, hidden 256, output 10)
  // uploading weight at 5090's VRAM
  int batch_size = 512;
  int max_epochs = 10;
  NeuralNetwork ann(784, 256, 10, batch_size);

  std::cout << "Starting calculation on 5090. (Num of data: " << image_count << ")" << std::endl;

  // 3. processing batch loop
  for(int epoch = 0; epoch < max_epochs; epoch++)
  {
    float epoch_loss = 0.0f;  // for checking error

    for(int i = 0; i < image_count; i += batch_size)
    {
      // calculate current batch size (handling last remained batch)
      int currnet_batch_size = std::min(batch_size, image_count - i);
      
      // 1. calculate fisrt image pointer in current batch
      const float* batch_start_ptr = &train_images[i * 784];

      // 2. copy it into GPU's d_input area (mem size: number of data *  784);
      ann.copy_to_device(batch_start_ptr, currnet_batch_size * 784);

      //TODO: 3. execute forward calculation
      //ann.forward(...);
      
      //TODO: 4. backward calculation (backpropagation and update weights)
      //ann.backward(...);
    }
    std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
  }

  std::cout<< "Done!" << std::endl;

  return 0;
}
