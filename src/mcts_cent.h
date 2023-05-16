#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include <gomoku.h>
#include <thread_pool.h>
#include <libtorch.h>

class TreeNode_cent {
 public:
  // friend class can access private variables
  friend class MCTS_cent;

  TreeNode_cent();
  TreeNode_cent(const TreeNode_cent &node);
  TreeNode_cent(TreeNode_cent *parent, double p_sa, unsigned action_size);

  TreeNode_cent &operator=(const TreeNode_cent &p);

  unsigned int select(double c_puct, double c_virtual_loss);
  void expand(const std::vector<double> &action_priors);
  void backup(double leaf_value);

  double get_value(double c_puct, double c_virtual_loss,
                   unsigned int sum_n_visited) const;
  inline bool get_is_leaf() const { return this->is_leaf; }

 private:
  // store tree
  TreeNode_cent *parent;
  std::vector<TreeNode_cent *> children;
  bool is_leaf;
  std::mutex lock;

  int n_visited;
  double p_sa;
  double q_sa;
  int virtual_loss;
};

class MCTS_cent {
 public:
  MCTS_cent(NeuralNetwork *neural_network, unsigned int thread_num, double c_puct,
       unsigned int num_mcts_sims, double c_virtual_loss,
       unsigned int action_size);
  std::vector<double> get_action_probs(Gomoku *gomoku, double temp = 1e-3);
  void update_with_move(int last_move);

 private:
  void simulate(Gomoku *gomoku);
  static void tree_deleter(TreeNode_cent *t);
  std::pair<TreeNode_cent *, std::pair<double, std::vector<double>>> exc_sim(TreeNode_cent* node, std::shared_ptr<Gomoku> game);

  // variables
  std::unique_ptr<TreeNode_cent, decltype(MCTS_cent::tree_deleter) *> root;
  std::unique_ptr<ThreadPool> thread_pool;
  NeuralNetwork *neural_network;

  unsigned int action_size;
  unsigned int num_mcts_sims;
  double c_puct;
  double c_virtual_loss;
};
