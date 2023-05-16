#include <libtorch.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>

using namespace std::chrono_literals;

NeuralNetwork::NeuralNetwork(std::string model_path, bool use_gpu,
                             unsigned int batch_size)
    : module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str()))),
      use_gpu(use_gpu),
      batch_size(batch_size),
      running(true),
      loop(nullptr)
{
  if (this->use_gpu)
  {
    // move to CUDA
    this->module->to(at::kCUDA);
  }
}

NeuralNetwork::~NeuralNetwork()
{
}

std::vector<std::vector<double>> NeuralNetwork::commit(Gomoku *gomoku)
{
  int n = gomoku->get_n();

  // convert data format
  auto board = gomoku->get_board();
  std::vector<int> board0;
  for (unsigned int i = 0; i < board.size(); i++)
  {
    board0.insert(board0.end(), board[i].begin(), board[i].end());
  }

  torch::Tensor temp =
      torch::from_blob(&board0[0], {1, 1, n, n}, torch::dtype(torch::kInt32));

  torch::Tensor state0 = temp.gt(0).toType(torch::kFloat32);
  torch::Tensor state1 = temp.lt(0).toType(torch::kFloat32);

  int last_move = gomoku->get_last_move();
  int cur_player = gomoku->get_current_color();

  if (cur_player == -1)
  {
    std::swap(state0, state1);
  }

  torch::Tensor state2 =
      torch::zeros({1, 1, n, n}, torch::dtype(torch::kFloat32));

  if (last_move != -1)
  {
    state2[0][0][last_move / n][last_move % n] = 1;
  }

  // torch::Tensor states = torch::cat({state0, state1}, 1);
  torch::Tensor states = torch::cat({state0, state1, state2}, 1);

  // infer
  at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();
  at::cuda::setCurrentCUDAStream(myStream);
  std::vector<torch::jit::IValue> inputs{
      this->use_gpu ? torch::cat(states, 0).to(at::kCUDA)
                    : torch::cat(states, 0)};
  auto result = this->module->forward(inputs).toTuple();

  torch::Tensor p_batch = result->elements()[0]
                              .toTensor()
                              .exp()
                              .toType(torch::kFloat32)
                              .to(at::kCPU);
  torch::Tensor v_batch =
      result->elements()[1].toTensor().toType(torch::kFloat32).to(at::kCPU);

  // set promise value
  torch::Tensor p = p_batch[0];
  torch::Tensor v = v_batch[0];

  std::vector<double> prob(static_cast<float *>(p.data_ptr()),
                           static_cast<float *>(p.data_ptr()) + p.size(0));
  std::vector<double> value{v.item<float>()};

  return {std::move(prob), std::move(value)};
}

// TODO: use lock-free queue
// https://github.com/cameron314/concurrentqueue
void NeuralNetwork::infer()
{
}
