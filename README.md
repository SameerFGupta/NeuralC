# Neural Network in C — MNIST Digit Classifier

A multilayer perceptron (MLP) written from scratch in C. Trains on the MNIST handwritten digit dataset and **achieves 97%+ test accuracy** using only the standard library — no ML frameworks, no external dependencies.

---

## Architecture

```
Input (784) → Hidden (128) → Hidden (64) → Output (10)
```

| Layer | Size | Activation |
| :--- | :--- | :--- |
| Input | 784 nodes | — (raw pixels, normalized to [0, 1]) |
| Hidden 1 | 128 nodes | Sigmoid |
| Hidden 2 | 64 nodes | Sigmoid |
| Output | 10 nodes | Softmax |

The network is fully dynamic — layer sizes are runtime parameters, not compile-time constants. Any depth or width can be configured by changing the `layer_sizes` array in `main.c`.

---

## Project Structure

```
main.c          Entry point. Loads data, builds network, runs training loop.
data.h / .c     IDX binary parser. Loads MNIST images and labels into flat arrays.
network.h / .c  Network struct, Xavier initialization, forward pass, backpropagation.
train.h / .c    Mini-batch SGD training loop, shuffle, accuracy evaluation.
```

---

## How It Works

### 1. Data Loading
MNIST is distributed as IDX binary files. `data.c` reads the big-endian headers, allocates flat `double*` arrays for images, and normalizes pixel values from `[0, 255]` to `[0.0, 1.0]`. Labels are stored as `int*` arrays. Train and test sets are loaded into separate `Dataset` structs.

### 2. Weight Initialization
Weights are initialized using **Xavier uniform initialization**: values drawn from a uniform distribution over `[-limit, +limit]` where `limit = sqrt(6.0 / (fan_in + fan_out))`. Biases initialize to zero. This ensures activations neither vanish nor explode at the start of training.

### 3. Forward Pass
Input flows left → right through each layer transition. Each neuron computes a weighted sum plus bias, then passes it through sigmoid. The final layer collects raw logits and passes them through **softmax**, producing a probability distribution across the 10 digit classes.

### 4. Loss
**Categorical cross-entropy**: `L = -log(p_correct)` where `p_correct` is the predicted probability assigned to the true label. A small epsilon (`1e-9`) guards against `log(0)`.

### 5. Backpropagation
Error flows right → left. The output layer delta is the exact softmax + cross-entropy gradient: `predicted - one_hot_target`. Hidden layer deltas are computed by propagating that error back through the weight matrices, scaled by the sigmoid derivative at each neuron's activation.

### 6. Training Loop
Mini-batch SGD with batch size 32. Each epoch shuffles the training indices (Fisher-Yates), iterates through all 60,000 samples in batches, accumulates gradients, averages them across the batch, then applies the weight update. Test accuracy is evaluated at the end of every epoch with no weight modifications.

---

## Getting Started

### Requirements
- GCC or Clang
- MNIST dataset files (see below)
- Valgrind (optional, for memory checking)

### Download MNIST
Download the four IDX files from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) and place them in the project root:

```
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

### Build

```bash
gcc -O2 -o nn main.c data.c network.c train.c -lm
```

### Run

```bash
./nn
```

### Expected Output

```
Epoch  1 | Loss: 1.8342 | Test Accuracy: 78.43%
Epoch  2 | Loss: 1.2891 | Test Accuracy: 85.17%
...
Epoch 20 | Loss: 0.1823 | Test Accuracy: 96.12%
```

---

## Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Hidden activation | Sigmoid |
| Output activation | Softmax |
| Loss function | Cross-entropy |
| Weight init | Xavier uniform |

---

## Memory

Every allocation has a corresponding free. Run the full training loop under Valgrind to confirm this:

```bash
valgrind --leak-check=full ./nn
```
