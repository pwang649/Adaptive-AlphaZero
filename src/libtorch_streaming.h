#pragma once

#include <torch/script.h>  // One-stop header.

#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <gomoku.h>

class NeuralNetwork {
 public:
  using return_type = std::vector<std::vector<double>>;

  NeuralNetwork(std::string model_path, bool use_gpu, unsigned int batch_size);
  ~NeuralNetwork();

  std::vector<std::vector<double>> commit(Gomoku* gomoku);
  void set_batch_size(unsigned int batch_size) {    // set batch_size
    this->batch_size = batch_size;
  };

  unsigned int get_inf_time() {
    return this->inf_time;
  }

  void set_inf_time(unsigned int t) {
	  this->inf_time = t;
  }

 private:
  using task_type = std::pair<torch::Tensor, std::promise<return_type>>;

  void infer();  // infer

  std::unique_ptr<std::thread> loop;  // call infer in loop
  bool running;                       // is running

  std::queue<task_type> tasks;  // tasks queue
  std::mutex lock;              // lock for tasks queue
  std::condition_variable cv;   // condition variable for tasks queue

  std::shared_ptr<torch::jit::script::Module> module;  // torch module
  unsigned int batch_size;                             // batch size
  bool use_gpu;                                        // use gpu
  unsigned int inf_time;
};
