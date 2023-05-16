# Multi-Core Acceleration of DNN-guided Tree-Parallel MCTS using Adaptive Parallelism

Two multi-threaded implementations of AlphaZero using [decentralized](https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf) and [centralized](https://arxiv.org/abs/1810.11755) tree-parallel MCTS.

## Getting Started

### Package Requirement

* CUDA 10+
* [LibTorch 1.1+ (Pre-cxx11 ABI)](https://pytorch.org/get-started/locally/)
* [SWIG 3.0.12+](https://sourceforge.net/projects/swig/files/)
* `requirements.txt`

### Compile and Run

```bash
# Compile Python extension
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=PATH/TO/LIBTORCH -DPYTHON_EXECUTABLE=PATH/TO/PYTHON -DCMAKE_BUILD_TYPE=Release
make -j10

# Run
cd ../test
python learner_test.py train # train model
python learner_test.py play  # play with human
```

## High-Level Structure

The code is organized as follows:

    .
    ├── config.py               # The central place to change parameters
    ├── src                     # Contains all c++ and Python source files
    │   ├── learner.py          # Python starter interface with c__
    │   ├── mcts_cent.*         # Centralized Implementation
    │   ├── mcts_decent.*       # Decentralized Implementation
    │   ├── libtorch.*          # Neural network task management
    │   ├── gomoku.*            # Game related
    │   └── ...              
    ├── test                    # Contains all Python test scripts
    │   ├── learner_test.py     # Main testing script for training and playing
    │   └── ...             
    └── ... 

## Reproducing Experiments

### [Data Sheet](https://docs.google.com/spreadsheets/d/1ah4ABjvCia2IQ4HzI3qgHpgjhnY7oYFXnCSYVJekD3E/edit?usp=sharing)

### Decentralized-tarim

* Experiment 1:

    ```python
    'centralized' : False,
    'libtorch_use_gpu' : False,
    'train_use_gpu' : False,
    'num_mcts_threads': ?, # Specify yourself
    ```

* Experiment 2:

    ```python
    'centralized' : False,
    'libtorch_use_gpu' : True,
    'train_use_gpu' : True,
    'num_mcts_threads': ?,      # Specify number of threads to use here
    ```

### Centralized-tarim

* Experiment 1:

    ```python
    'centralized' : True,
    'libtorch_use_gpu' : False,
    'train_use_gpu' : False,
    'num_mcts_threads': ?,      # Specify number of threads to use here
    ```

* Experiment 2:

    ```python
    'centralized' : True,
    'libtorch_use_gpu' : True,
    'train_use_gpu' : True,
    'num_mcts_threads': ?,      # Specify number of threads to use here
    'inference_batch_size': ?,  # Specify batch size for inference here
    ```

    > Optimize `inference_batch_size` when running the centralized method on GPU. Use `-1` on other occasions.

* Experiment 3:

    ```python
    'centralized' : True,
    'libtorch_use_gpu' : True,
    'train_use_gpu' : True,
    'num_mcts_threads': 64,     # We set N=64 and explore the effect of different batch sizes
    'inference_batch_size': ?,  # Specify batch size for inference here
    ```

* Experiment 4:

    ```python
    'centralized' : True,
    'libtorch_use_gpu' : True,
    'train_use_gpu' : True,
    'num_mcts_threads': 32,     # Same as Experiment 3, but with N=32
    'inference_batch_size': ?,  # Specify batch size for inference here
    ```

* Experiment 5:

  * Same as Experiment 3, but it measures the total inference time instead of the total run time.
  * Uncomment lines 122, 124-126 in `libtorch.cpp` and lines 334, 405 in `mcts_cent.cpp` to enable the inference time profiling feature.

## References

1. Mastering the Game of Go without Human Knowledge
2. Mastering atari, go, chess and shogi by planning with a learned model
3. Parallel Monte-Carlo Tree Search
4. An Analysis of Virtual Loss in Parallel MCTS
5. A hybrid gomoku deep learning artificial intelligence
6. <https://github.com/hijkzzz/alpha-zero-gomoku>
