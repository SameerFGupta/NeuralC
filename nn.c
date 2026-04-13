#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_INPUTS 2
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 1

double init_weight() {
    return ((double)rand()) / ((double)RAND_MAX);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x) {
    return x * (1.0 - x);
}

void forward_pass(double inputs[NUM_INPUTS],
                  double hidden_weights[NUM_INPUTS][NUM_HIDDEN], double hidden_biases[NUM_HIDDEN],
                  double output_weights[NUM_HIDDEN][NUM_OUTPUTS], double output_biases[NUM_OUTPUTS],
                  double hidden_layer[NUM_HIDDEN], double output_layer[NUM_OUTPUTS]) {

    for (int j = 0; j < NUM_HIDDEN; j++) {
        double activation = hidden_biases[j];
        for (int i = 0; i < NUM_INPUTS; i++) {
            activation += inputs[i] * hidden_weights[i][j];
        }
        hidden_layer[j] = sigmoid(activation);
    }

    for (int j = 0; j < NUM_OUTPUTS; j++) {
        double activation = output_biases[j];
        for (int i = 0; i < NUM_HIDDEN; i++) {
            activation += hidden_layer[i] * output_weights[i][j];
        }
        output_layer[j] = sigmoid(activation);
    }
}

void backpropagate(double target, double output_layer[NUM_OUTPUTS], double hidden_layer[NUM_HIDDEN], double output_weights[NUM_HIDDEN][NUM_OUTPUTS], double output_delta[NUM_OUTPUTS], double hidden_deltas[NUM_HIDDEN]) {
    output_delta[0] = (target - output_layer[0]) * dSigmoid(output_layer[0]);

    for (int i = 0; i < NUM_HIDDEN; i++) {
        hidden_deltas[i] = (output_delta[0] * output_weights[i][0]) * dSigmoid(hidden_layer[i]);
    }
}

void update_weights(double learning_rate, double inputs[NUM_INPUTS], double hidden_layer[NUM_HIDDEN], double output_delta[NUM_OUTPUTS], double hidden_deltas[NUM_HIDDEN], double hidden_weights[NUM_INPUTS][NUM_HIDDEN], double hidden_biases[NUM_HIDDEN], double output_weights[NUM_HIDDEN][NUM_OUTPUTS], double output_biases[NUM_OUTPUTS]) {
    for (int j = 0; j < NUM_OUTPUTS; j++) {
        output_biases[j] += learning_rate * output_delta[j];
        for (int i = 0; i < NUM_HIDDEN; i++) {
            output_weights[i][j] += learning_rate * output_delta[j] * hidden_layer[i];
        }
    }

    for (int j = 0; j < NUM_HIDDEN; j++) {
        hidden_biases[j] += learning_rate * hidden_deltas[j];
        for (int i = 0; i < NUM_INPUTS; i++) {
            hidden_weights[i][j] += learning_rate * hidden_deltas[j] * inputs[i];
        }
    }
}

int main(void) {
    double hidden_weights[NUM_INPUTS][NUM_HIDDEN];
    double hidden_biases[NUM_HIDDEN];
    double output_weights[NUM_HIDDEN][NUM_OUTPUTS];
    double output_biases[NUM_OUTPUTS];

    for (int i = 0; i < NUM_INPUTS; i++) {
        for (int j = 0; j < NUM_HIDDEN; j++) {
            hidden_weights[i][j] = init_weight();
        }
    }

    for (int i = 0; i < NUM_HIDDEN; i++) {
        hidden_biases[i] = init_weight();
    }

    for (int i = 0; i < NUM_HIDDEN; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            output_weights[i][j] = init_weight();
        }
    }

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        output_biases[i] = init_weight();
    }

    return 0;
}
