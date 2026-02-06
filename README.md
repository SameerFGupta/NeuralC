Simple Neural Network in C
This is a basic neural network written from scratch in C. It solves the classic XOR problem (exclusive OR), which is a "Hello World" benchmark for neural networks because it requires the network to learn non-linear patterns.
This program implements a Multilayer Perceptron (MLP). It learns to predict outputs by adjusting internal numbers called weights over thousands of attempts.


1. The Architecture
The network has a 2-2-1 layout:

Input Layer (2 nodes): Accepts two binary inputs (0 or 1).

Hidden Layer (2 nodes): The "brain" of the network that allows it to solve complex problems.

Output Layer (1 node): Produces the final probability (closer to 0 or 1).

2. The Process (The Loop)
The training process happens in a massive loop (100,000 times). Inside that loop, two main things happen for every data point:

Forward Pass (Prediction): The data flows from Left -> Right.

Inputs are multiplied by weights and added to biases.

The result is squashed by the Sigmoid Activation Function to keep values between 0 and 1.

This happens for the hidden layer, then again for the output layer.

Backpropagation: The error flows from Right -> Left.

We verify the output against the correct answer (e.g., Input: 0, 1 -> Expected: 1).

We calculate the Error (how wrong were we?).

Using calculus (chain rule), we calculate how much each specific weight contributed to that error.

We nudge the weights slightly in the opposite direction to reduce the error next time.

3. The Data (XOR)
The network trains on these four cases: | Input A | Input B | Target | | :--- | :--- | :--- | | 0 | 0 | 0 | | 0 | 1 | 1 | | 1 | 0 | 1 | | 1 | 1 | 0 |
