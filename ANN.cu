#include "ANN.h"
#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <curand.h>
#include <iterator>
#include <vector>


__global__ void sigmoidKernel(float* data, int size)
{
  // In the GPU, calculate idx of 1D that this thread will handle
  int workIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // Check 
  if(workIdx < size)
  {
    data[workIdx] = 1.0f / (1.0f + expf(-data[workIdx]));
  }
}


/*
 * gpuErrchk
 * -->
 * */
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, 
                      const char *file, 
                      int line, 
                      bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

NeuralNetwork::NeuralNetwork(int in, int hidden, int out, int batch_size)
  : input_size(in), hidden_size(hidden), output_size(out), max_batch_size(batch_size)
{
  // 1. allocate weights and bias
  gpuErrchk(cudaMalloc(&this->d_W1, in * hidden * sizeof(float)));
  gpuErrchk(cudaMalloc(&this->d_b1, hidden * sizeof(float)));
  gpuErrchk(cudaMalloc(&this->d_W2, hidden * out * sizeof(float)));
  gpuErrchk(cudaMalloc(&this->d_b2, out * sizeof(float)));

  // 2. allocate space for activated value layer
  gpuErrchk(cudaMalloc(&this->d_input,  batch_size * in * sizeof(float)));
  gpuErrchk(cudaMalloc(&this->d_hidden, batch_size * hidden * sizeof(float)));
  gpuErrchk(cudaMalloc(&this->d_output, batch_size * out * sizeof(float)));

  std::cout << "Successfully allocated memory on RTX 5090." << std::endl;

  //TODO: add initializing weights logics (USE curand)

  // 3. initializing weights with cuRAND
  curandGenerator_t gen;
  // making random number generator
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  // set seed (using 1234ULL)
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

  //initializing W1 (random number within avg = 0.0, stand_deviation = 0.05)
  curandGenerateNormal(gen, this->d_W1, in * hidden, 0.0f, 0.05f);
  // initializing W2
  curandGenerateNormal(gen, this->d_W2, in * out, 0.0f, 0.05f);

  // bias initialize (0)
  gpuErrchk(cudaMemset(this->d_b1, 0, hidden * sizeof(float)));
  gpuErrchk(cudaMemset(this->d_b2, 0, out * sizeof(float)));

  // free the generator
  curandDestroyGenerator(gen);

  cublasCreate(&this->cublas_handle);
}

NeuralNetwork::~NeuralNetwork()
{
  // preventing memory leak
  cudaFree(this->d_W1);
  cudaFree(this->d_b1);
  cudaFree(this->d_W2);
  cudaFree(this->d_b2);
  cudaFree(this->d_input);
  cudaFree(this->d_hidden);
  cudaFree(this->d_output);

  cublasDestroy(this->cublas_handle);
}

void NeuralNetwork::copy_to_device(const float* host_data, int size) 
{
    gpuErrchk(cudaMemcpy(this->d_input, host_data, size * sizeof(float), cudaMemcpyHostToDevice));
}

void NeuralNetwork::forward(const std::vector<float>& h_input)
{
  // 1. copy host data to GPU (using copy_to_device())
  // assuming it will get each batch size(curr_batch_size) from the main
  // void forward(const float* d_batch_input, int curr_batch_size) will be ideal?

  int row     = this->max_batch_size; //512
  int column  = this->hidden_size;    //256
  int k       = this->input_size;     // middle demension 784

  float alpha = 1.0f;
  float beta  = 0.0f;

  // - {Layer 1} Input (512 * 784) * W1 (784*256) = Hidden (512 * 256) -
  // since cuBLAS is based on Column, to do Row-Major matrix multiplication, it needs order and transposition setting like below
  cublasSgemm(this->cublas_handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              column, row, k,
              &alpha,
              this->d_W1, column,
              this->d_input, k,
              &beta,
              this->d_hidden, row);

  // TODO: bias(d_b1) addition needs to be added
  
  // sigmoid function 
  int total_hidden_elements = row * column; // 512 * 256
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_hidden_elements + threadsPerBlock - 1) /threadsPerBlock;

  sigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(this->d_hidden, total_hidden_elements);

  // -{layer 2} Hidden (512x256) * W2 (256x10) = Output (512x10) -
  int out_n = output_size; // 10
  int out_k = hidden_size; // 256
  
  cublasSgemm(this->cublas_handle, 
              CUBLAS_OP_N, CUBLAS_OP_N, 
              out_n, row, out_k, 
              &alpha, 
              this->d_W2, out_n, 
              this->d_hidden, out_k, 
              &beta, 
              this->d_output, out_n);
  // TODO: bias(d_b2) addition needs to be added
  //
  //
  int total_output_elements = row * out_n; // 512 * 10
  blocksPerGrid = (total_output_elements + threadsPerBlock - 1) / threadsPerBlock;
    
  sigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, total_output_elements);
} 
