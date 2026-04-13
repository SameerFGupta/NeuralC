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
