#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

class NeuralNetwork 
{
  private:
    // layer size info
    int input_size, hidden_size, output_size;
    int max_batch_size;
    float learning_rate = 0.01f;

    // Device(GPU) weight pointer
    float *d_W1, *d_b1; // input -> hidden
    float *d_W2, *d_b2; // hidden -> input

    // activation value of each layer
    float *d_input, *d_hidden, *d_output;

    cublasHandle_t cublas_handle;

  public:
    NeuralNetwork(int in, int hidden, int out, int batch_size);
    ~NeuralNetwork();
    
    void forward(const std::vector<float>& h_input);
    std::vector<float> get_output();
    
    // updating weights (backpropagation)
    void backward(const std::vector<float>& h_labels);

    // save and load trained weight (for inference)
    void save_weights(std::string path);
    void load_weights(std::string path);

    // copying batch data func for optimization (5090)
    void copy_to_device(const float* host_data, int size);
};
