# Neural Network in C — MNIST Digit Classifier

A multilayer perceptron (MLP) written from scratch in C. Trains on the MNIST handwritten digit dataset and **achieves 97%+ test accuracy** using only the standard library — no ML frameworks, no external dependencies, no Python.

For context: the original 1998 LeCun paper that introduced MNIST reported 95.3% with a basic network. This beats it.

---

## Results

| Epoch | Loss | Test Accuracy |
| :--- | :--- | :--- |
| 1 | 0.3236 | 90.94% |
| 5 | 0.1675 | 95.06% |
| 10 | 0.1041 | 96.74% |
| 16 | 0.0675 | **97.44%** |

Accuracy crosses 95% by epoch 5. Crosses 97% by epoch 16. Loss decreases monotonically across all epochs with no instability.

---

## Architecture

```
Input (784) → Hidden (128) → Hidden (64) → Output (10)
```

| Layer | Size | Activation |
| :--- | :--- | :--- |
| Input | 784 nodes | — (raw pixels, normalized to [0, 1]) |
| Hidden 1 | 128 nodes | ReLU |
| Hidden 2 | 64 nodes | ReLU |
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
Input flows left → right through each layer transition. Each neuron computes a weighted sum plus bias, then passes it through ReLU. The final layer collects raw logits and passes them through **softmax**, producing a probability distribution across the 10 digit classes.

### 4. Loss
**Categorical cross-entropy**: `L = -log(p_correct)` where `p_correct` is the predicted probability assigned to the true label. A small epsilon (`1e-9`) guards against `log(0)`.

### 5. Backpropagation
Error flows right → left. The output layer delta is the exact softmax + cross-entropy gradient: `predicted - one_hot_target`. Hidden layer deltas are computed by propagating that error back through the weight matrices, scaled by the ReLU derivative at each neuron's activation.

### 6. Training Loop
Mini-batch SGD with batch size 32. Each epoch shuffles the training indices (Fisher-Yates), iterates through all 60,000 samples in batches, accumulates gradients, averages them across the batch, then applies the weight update. Test accuracy is evaluated at the end of every epoch with no weight modifications.

---

## Getting Started

### Requirements
- GCC or Clang
- MNIST dataset files (see below)

### Download MNIST
Download the four IDX files from the GitHub mirror and place them in the project root:

```bash
wget https://github.com/cvdfoundation/mnist/raw/main/train-images-idx3-ubyte.gz
wget https://github.com/cvdfoundation/mnist/raw/main/train-labels-idx1-ubyte.gz
wget https://github.com/cvdfoundation/mnist/raw/main/t10k-images-idx3-ubyte.gz
wget https://github.com/cvdfoundation/mnist/raw/main/t10k-labels-idx1-ubyte.gz

gunzip *.gz
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
Epoch  1 | Loss: 0.3236 | Test Accuracy: 90.94%
Epoch  5 | Loss: 0.1675 | Test Accuracy: 95.06%
Epoch 10 | Loss: 0.1041 | Test Accuracy: 96.74%
Epoch 16 | Loss: 0.0675 | Test Accuracy: 97.44%
```

---

## Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Epochs | 40 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Hidden activation | ReLU |
| Output activation | Softmax |
| Loss function | Cross-entropy |
| Weight init | Xavier uniform |

---

## Memory

Every allocation has a corresponding free. Verified clean under AddressSanitizer:

```bash
gcc -O2 -Wall -Wextra -fsanitize=address -o nn main.c data.c network.c train.c -lm
./nn
```

Expected: zero errors, zero leaks.
