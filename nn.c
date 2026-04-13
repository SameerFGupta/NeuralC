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
