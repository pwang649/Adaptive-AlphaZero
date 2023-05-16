#include <math.h>
#include <float.h>
#include <chrono>
#include <numeric>
#include <iostream>

#include <mcts.h>

using namespace std::chrono;

// TreeNode
TreeNode::TreeNode()
    : parent(nullptr),
      is_leaf(true),
      virtual_loss(0),
      n_visited(0),
      p_sa(0),
      q_sa(0) {}

TreeNode::TreeNode(TreeNode *parent, double p_sa, unsigned int action_size)
    : parent(parent),
      children(action_size, nullptr),
      is_leaf(true),
      virtual_loss(0),
      n_visited(0),
      q_sa(0),
      p_sa(p_sa) {}

TreeNode::TreeNode(
    const TreeNode &node)
{ // because automic<>, define copy function
  // struct
  this->parent = node.parent;
  this->children = node.children;
  this->is_leaf = node.is_leaf;

  this->n_visited = node.n_visited;
  this->p_sa = node.p_sa;
  this->q_sa = node.q_sa;

  this->virtual_loss = node.virtual_loss;
}

TreeNode &TreeNode::operator=(const TreeNode &node)
{
  if (this == &node)
  {
    return *this;
  }

  // struct
  this->parent = node.parent;
  this->children = node.children;
  this->is_leaf = node.is_leaf;

  this->n_visited = node.n_visited;
  this->p_sa = node.p_sa;
  this->q_sa = node.q_sa;
  this->virtual_loss = node.virtual_loss;

  return *this;
}

unsigned int TreeNode::select(double c_puct, double c_virtual_loss)
{
  double best_value = -DBL_MAX;
  unsigned int best_move = 0;
  TreeNode *best_node;

  for (unsigned int i = 0; i < this->children.size(); i++)
  {
    // empty node
    if (children[i] == nullptr)
    {
      continue;
    }

    unsigned int sum_n_visited = this->n_visited + 1;
    double cur_value =
        children[i]->get_value(c_puct, c_virtual_loss, sum_n_visited);
    if (cur_value > best_value)
    {
      best_value = cur_value;
      best_move = i;
      best_node = children[i];
    }
  }

  // add vitural loss
  best_node->virtual_loss++;

  return best_move;
}

void TreeNode::expand(const std::vector<double> &action_priors)
{
  {
    // get lock
    // std::lock_guard<std::mutex> lock(this->lock);

    if (this->is_leaf)
    {
      unsigned int action_size = this->children.size();

      for (unsigned int i = 0; i < action_size; i++)
      {
        // illegal action
        if (abs(action_priors[i] - 0) < FLT_EPSILON)
        {
          continue;
        }
        this->children[i] = new TreeNode(this, action_priors[i], action_size);
      }

      // not leaf
      this->is_leaf = false;
    }
  }
}

void TreeNode::backup(double value)
{
  // If it is not root, this node's parent should be updated first
  if (this->parent != nullptr)
  {
    this->parent->backup(-value);
  }

  // remove vitural loss
  this->virtual_loss--;

  // update n_visited
  unsigned int n_visited = this->n_visited;
  this->n_visited++;

  // update q_sa
  {
    this->q_sa = (n_visited * this->q_sa + value) / (n_visited + 1);
  }
}

double TreeNode::get_value(double c_puct, double c_virtual_loss,
                           unsigned int sum_n_visited) const
{
  // u
  auto n_visited = this->n_visited;
  double u = (c_puct * this->p_sa * sqrt(sum_n_visited) / (1 + n_visited));

  // virtual loss
  double virtual_loss = c_virtual_loss * this->virtual_loss;
  // int n_visited_with_loss = n_visited - virtual_loss;

  if (n_visited <= 0)
  {
    return u;
  }
  else
  {
    return u + (this->q_sa * n_visited - virtual_loss) / n_visited;
  }
}

// MCTS
MCTS::MCTS(NeuralNetwork *neural_network, unsigned int thread_num, double c_puct,
           unsigned int num_mcts_sims, double c_virtual_loss,
           unsigned int action_size)
    : neural_network(neural_network),
      thread_pool(new ThreadPool(thread_num)),
      c_puct(c_puct),
      num_mcts_sims(num_mcts_sims),
      c_virtual_loss(c_virtual_loss),
      action_size(action_size),
      root(new TreeNode(nullptr, 1., action_size), MCTS::tree_deleter) {}

void MCTS::update_with_move(int last_action)
{
  auto old_root = this->root.get();

  // reuse the child tree
  if (last_action >= 0 && old_root->children[last_action] != nullptr)
  {
    // unlink
    TreeNode *new_node = old_root->children[last_action];
    old_root->children[last_action] = nullptr;
    new_node->parent = nullptr;

    this->root.reset(new_node);
  }
  else
  {
    this->root.reset(new TreeNode(nullptr, 1., this->action_size));
  }
}

void MCTS::tree_deleter(TreeNode *t)
{
  if (t == nullptr)
  {
    return;
  }

  // remove children
  for (unsigned int i = 0; i < t->children.size(); i++)
  {
    if (t->children[i])
    {
      tree_deleter(t->children[i]);
    }
  }

  // remove self
  delete t;
}

std::vector<double> MCTS::get_action_probs(Gomoku *gomoku, double temp)
{
  auto begin = high_resolution_clock::now();

  simulate(gomoku);

  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end - begin);
  std::cout << "Time for on Run: " << duration.count() << " us." << std::endl;

  // calculate probs
  std::vector<double> action_probs(gomoku->get_action_size(), 0);
  const auto &children = this->root->children;

  // greedy
  if (temp - 1e-3 < FLT_EPSILON)
  {
    unsigned int max_count = 0;
    unsigned int best_action = 0;

    for (unsigned int i = 0; i < children.size(); i++)
    {
      if (children[i] && children[i]->n_visited > max_count)
      {
        max_count = children[i]->n_visited;
        best_action = i;
      }
    }

    action_probs[best_action] = 1.;
    return action_probs;
  }
  else
  {
    // explore
    double sum = 0;
    for (unsigned int i = 0; i < children.size(); i++)
    {
      if (children[i] && children[i]->n_visited > 0)
      {
        action_probs[i] = pow(children[i]->n_visited, 1 / temp);
        sum += action_probs[i];
      }
    }

    // renormalization
    std::for_each(action_probs.begin(), action_probs.end(),
                  [sum](double &x)
                  { x /= sum; });

    return action_probs;
  }
}

template <typename R>
bool future_is_ready(std::future<R> const &f)
{
  return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// expansion + simulation
std::pair<TreeNode *, std::pair<double, std::vector<double>>> MCTS::exc_sim(TreeNode *node, std::shared_ptr<Gomoku> game)
{
  std::vector<double> action_priors(this->action_size, 0);

  std::vector<std::vector<double>> result = this->neural_network->commit(game.get());

  action_priors = std::move(result[0]);
  double value = result[1][0];

  // mask invalid actions
  auto legal_moves = game->get_legal_moves();
  double sum = 0;
  for (unsigned int i = 0; i < action_priors.size(); i++)
  {
    if (legal_moves[i] == 1)
    {
      sum += action_priors[i];
    }
    else
    {
      action_priors[i] = 0;
    }
  }

  // renormalization
  if (sum > FLT_EPSILON)
  {
    std::for_each(action_priors.begin(), action_priors.end(),
                  [sum](double &x)
                  { x /= sum; });
  }
  else
  {
    std::cout << "All valid moves were masked, do workaround." << std::endl;

    sum = std::accumulate(legal_moves.begin(), legal_moves.end(), 0);
    for (unsigned int i = 0; i < action_priors.size(); i++)
    {
      action_priors[i] = legal_moves[i] / sum;
    }
  }

  return std::make_pair(node, std::make_pair(value, action_priors));
}

void MCTS::simulate(Gomoku *gomoku)
{

  std::vector<std::future<std::pair<TreeNode *, std::pair<double, std::vector<double>>>>> futures(this->thread_pool->get_idl_num());
  int completed = 0;
  int occupied = 0;
  int available = 0;

  while (completed < this->num_mcts_sims)
  {
    auto game = std::make_shared<Gomoku>(*gomoku);
    auto node = this->root.get();

    while (true)
    {
      if (node->get_is_leaf())
      {
        break;
      }

      // select
      auto action = node->select(this->c_puct, this->c_virtual_loss);
      game->execute_move(action);
      node = node->children[action];
    }

    auto status = game->get_game_status();
    double value = 0;

    // end state
    if (status[0] != 0)
    {
      auto winner = status[1];
      value = (winner == 0 ? 0 : (winner == game->get_current_color() ? 1 : -1));
      node->backup(-value);
      continue;
    }
    else
    {
      auto future = this->thread_pool->commit(std::bind(&MCTS::exc_sim, this, node, game));
      futures[available] = std::move(future);
      occupied++;
      available++;
      // }
      if (occupied >= futures.size())
      {
        // wait for one thread to finish
        unsigned int j = 0;
        while (true)
        {
          for (j = 0; j < futures.size(); j++)
          {
            if (future_is_ready(futures[j]))
            {
              auto result = futures[j].get();
              // expand
              result.first->expand(result.second.second);
              result.first->backup(-result.second.first);
              completed++;
              occupied--;
              available = j;
              break;
            }
          }
          if (j < futures.size())
          {
            break;
          }
        }
      }
    }
  }
  return;
}
