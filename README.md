# AlphaZero Gomoku

Two multi-threaded implementations of AlphaZero using [decentralized](https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf) and [centralized](https://arxiv.org/abs/1810.11755) tree-parallel MCTS.

## Branches

1. parallel: decentralized implementation
2. centralized: centralized implementation

## Features

* Easy Freestyle Gomoku
* Multi-threading Tree/Root Parallelization with Virtual Loss and LibTorch
* Gomoku, MCTS and Network Infer are written in C++
* SWIG for Python C++ extension

## Packages

* CUDA 10+
* [PyTorch 1.1+](https://pytorch.org/get-started/locally/)
* [LibTorch 1.1+ (Pre-cxx11 ABI)](https://pytorch.org/get-started/locally/)
* [SWIG 3.0.12+](https://sourceforge.net/projects/swig/files/)
* CMake 3.8+
* MSVC14.0+ / GCC6.0+

## config.py

```python
'inference_batch_size': -1,                 # -1: default to be num_mcts_thread
```

Optimize `inference_batch_size` when running the centralized method on GPU. Use `-1` on other occasions.

## Run

```bash
# Compile Python extension
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=path/to/libtorch -DPYTHON_EXECUTABLE=path/to/python -DCMAKE_BUILD_TYPE=Release
make -j10

# Run
cd ../test
python learner_test.py train # train model
python learner_test.py play  # play with human
```

## GUI

![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/assets/gomoku_gui.png)

## References

1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. Parallel Monte-Carlo Tree Search
4. An Analysis of Virtual Loss in Parallel MCTS
5. A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
6. https://github.com/hijkzzz/alpha-zero-gomoku