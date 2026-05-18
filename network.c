#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "network.h"

static double relu(double x) {
    if (x > 0.0) {
        return x;
    }

    return 0.0;
}

static double drelu(double activated_value) {
    if (activated_value > 0.0) {
        return 1.0;
    }

    return 0.0;
}

static void softmax(double* values, int n) {
    double max_value;
    double sum;
    int i;

    if (values == NULL || n <= 0) {
        return;
    }

    max_value = values[0];
    for (i = 1; i < n; i++) {
        if (values[i] > max_value) {
            max_value = values[i];
        }
    }

    sum = 0.0;
    for (i = 0; i < n; i++) {
        values[i] = exp(values[i] - max_value);
        sum += values[i];
    }

    if (sum == 0.0) {
        return;
    }

    for (i = 0; i < n; i++) {
        values[i] /= sum;
    }
}

double** alloc_matrix(int rows, int cols) {
    double* data;
    double** raw_ptrs;
    double** matrix;
    int i;

    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "alloc_matrix: invalid dimensions %d x %d\n", rows, cols);
        return NULL;
    }

    data = (double*) calloc((size_t) rows * (size_t) cols, sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "alloc_matrix: failed to allocate data block\n");
        return NULL;
    }

    raw_ptrs = (double**) malloc(((size_t) rows + 1U) * sizeof(double*));
    if (raw_ptrs == NULL) {
        fprintf(stderr, "alloc_matrix: failed to allocate row pointers\n");
        free(data);
        return NULL;
    }

    raw_ptrs[0] = data;
    matrix = raw_ptrs + 1;

    for (i = 0; i < rows; i++) {
        matrix[i] = data + ((size_t) i * (size_t) cols);
    }

    return matrix;
}

void free_matrix(double** matrix, int rows) {
    double** raw_ptrs;

    (void) rows;

    if (matrix == NULL) {
        return;
    }

    raw_ptrs = matrix - 1;
    free(raw_ptrs[0]);
    free(raw_ptrs);
}

static void free_layer_arrays(double** arrays, int n_layers) {
    int l;

    if (arrays == NULL) {
        return;
    }

    for (l = 0; l < n_layers; l++) {
        free(arrays[l]);
    }

    free(arrays);
}

void network_free(Network* net) {
    int l;

    if (net == NULL) {
        return;
    }

    if (net->weights != NULL) {
        for (l = 0; l < net->n_layers - 1; l++) {
            free_matrix(net->weights[l], net->layer_sizes[l]);
        }
        free(net->weights);
    }

    if (net->biases != NULL) {
        for (l = 0; l < net->n_layers - 1; l++) {
            free(net->biases[l]);
        }
        free(net->biases);
    }

    if (net->weight_grads != NULL) {
        for (l = 0; l < net->n_layers - 1; l++) {
            free_matrix(net->weight_grads[l], net->layer_sizes[l]);
        }
        free(net->weight_grads);
    }

    if (net->bias_grads != NULL) {
        for (l = 0; l < net->n_layers - 1; l++) {
            free(net->bias_grads[l]);
        }
        free(net->bias_grads);
    }

    free_layer_arrays(net->activations, net->n_layers);
    free_layer_arrays(net->deltas, net->n_layers);
    free(net->layer_sizes);
    free(net);
}

Network* network_create(int* layer_sizes, int n_layers) {
    Network* net;
    int l;

    if (layer_sizes == NULL || n_layers < 2) {
        fprintf(stderr, "network_create: invalid network shape\n");
        return NULL;
    }

    net = (Network*) calloc(1, sizeof(Network));
    if (net == NULL) {
        fprintf(stderr, "network_create: failed to allocate Network\n");
        return NULL;
    }

    net->n_layers = n_layers;
    net->layer_sizes = (int*) malloc((size_t) n_layers * sizeof(int));
    if (net->layer_sizes == NULL) {
        fprintf(stderr, "network_create: failed to allocate layer sizes\n");
        network_free(net);
        return NULL;
    }
    memcpy(net->layer_sizes, layer_sizes, (size_t) n_layers * sizeof(int));

    net->weights = (double***) calloc((size_t) (n_layers - 1), sizeof(double**));
    net->biases = (double**) calloc((size_t) (n_layers - 1), sizeof(double*));
    net->weight_grads = (double***) calloc((size_t) (n_layers - 1), sizeof(double**));
    net->bias_grads = (double**) calloc((size_t) (n_layers - 1), sizeof(double*));
    net->activations = (double**) calloc((size_t) n_layers, sizeof(double*));
    net->deltas = (double**) calloc((size_t) n_layers, sizeof(double*));
    if (net->weights == NULL || net->biases == NULL || net->weight_grads == NULL ||
        net->bias_grads == NULL || net->activations == NULL || net->deltas == NULL) {
        fprintf(stderr, "network_create: failed to allocate network arrays\n");
        network_free(net);
        return NULL;
    }

    for (l = 0; l < n_layers - 1; l++) {
        net->weights[l] = alloc_matrix(net->layer_sizes[l], net->layer_sizes[l + 1]);
        if (net->weights[l] == NULL) {
            network_free(net);
            return NULL;
        }

        net->weight_grads[l] = alloc_matrix(net->layer_sizes[l], net->layer_sizes[l + 1]);
        if (net->weight_grads[l] == NULL) {
            network_free(net);
            return NULL;
        }

        net->biases[l] = (double*) calloc((size_t) net->layer_sizes[l + 1], sizeof(double));
        net->bias_grads[l] = (double*) calloc((size_t) net->layer_sizes[l + 1], sizeof(double));
        if (net->biases[l] == NULL || net->bias_grads[l] == NULL) {
            fprintf(stderr, "network_create: failed to allocate biases or gradients for layer %d\n", l);
            network_free(net);
            return NULL;
        }
    }

    for (l = 0; l < n_layers; l++) {
        net->activations[l] = (double*) calloc((size_t) net->layer_sizes[l], sizeof(double));
        net->deltas[l] = (double*) calloc((size_t) net->layer_sizes[l], sizeof(double));
        if (net->activations[l] == NULL || net->deltas[l] == NULL) {
            fprintf(stderr, "network_create: failed to allocate layer buffers for layer %d\n", l);
            network_free(net);
            return NULL;
        }
    }

    return net;
}

void network_init_xavier(Network* net) {
    static int seeded = 0;
    int l;

    if (net == NULL) {
        fprintf(stderr, "network_init_xavier: net is NULL\n");
        return;
    }

    if (!seeded) {
        srand((unsigned int) time(NULL));
        seeded = 1;
    }

    for (l = 0; l < net->n_layers - 1; l++) {
        int fan_in = net->layer_sizes[l];
        int fan_out = net->layer_sizes[l + 1];
        double limit = sqrt(6.0 / (double) (fan_in + fan_out));
        int i;
        int j;

        for (i = 0; i < fan_in; i++) {
            for (j = 0; j < fan_out; j++) {
                net->weights[l][i][j] =
                    ((double) rand() / (double) RAND_MAX) * 2.0 * limit - limit;
            }
        }

        for (j = 0; j < fan_out; j++) {
            net->biases[l][j] = 0.0;
        }
    }
}

void network_forward(Network* net, double* input) {
    int l;
    int i;

    if (net == NULL || input == NULL) {
        fprintf(stderr, "network_forward: invalid input\n");
        return;
    }

    for (i = 0; i < net->layer_sizes[0]; i++) {
        net->activations[0][i] = input[i];
    }

    for (l = 0; l < net->n_layers - 1; l++) {
        int current_size = net->layer_sizes[l];
        int next_size = net->layer_sizes[l + 1];
        int j;

        for (j = 0; j < next_size; j++) {
            double sum = net->biases[l][j];

            for (i = 0; i < current_size; i++) {
                sum += net->activations[l][i] * net->weights[l][i][j];
            }

            if (l < net->n_layers - 2) {
                net->activations[l + 1][j] = relu(sum);
            } else {
                net->activations[l + 1][j] = sum;
            }
        }
    }

    softmax(
        net->activations[net->n_layers - 1],
        net->layer_sizes[net->n_layers - 1]
    );
}

double network_cross_entropy_loss(Network* net, int true_label) {
    double p;
    int output_size;

    if (net == NULL) {
        fprintf(stderr, "network_cross_entropy_loss: net is NULL\n");
        return 0.0;
    }

    output_size = net->layer_sizes[net->n_layers - 1];
    if (true_label < 0 || true_label >= output_size) {
        fprintf(stderr, "network_cross_entropy_loss: invalid true_label %d\n", true_label);
        return 0.0;
    }

    p = net->activations[net->n_layers - 1][true_label];
    return -log(p + 1e-9);
}

void network_output_deltas(Network* net, int true_label) {
    int output_layer;
    int output_size;
    int j;

    if (net == NULL) {
        fprintf(stderr, "network_output_deltas: net is NULL\n");
        return;
    }

    output_layer = net->n_layers - 1;
    output_size = net->layer_sizes[output_layer];
    if (true_label < 0 || true_label >= output_size) {
        fprintf(stderr, "network_output_deltas: invalid true_label %d\n", true_label);
        return;
    }

    for (j = 0; j < output_size; j++) {
        double one_hot = (j == true_label) ? 1.0 : 0.0;
        net->deltas[output_layer][j] = net->activations[output_layer][j] - one_hot;
    }
}

void network_backward(Network* net) {
    int l;

    if (net == NULL) {
        fprintf(stderr, "network_backward: net is NULL\n");
        return;
    }

    for (l = net->n_layers - 2; l >= 1; l--) {
        int current_size = net->layer_sizes[l];
        int next_size = net->layer_sizes[l + 1];
        int i;

        for (i = 0; i < current_size; i++) {
            double error = 0.0;
            int j;

            for (j = 0; j < next_size; j++) {
                error += net->deltas[l + 1][j] * net->weights[l][i][j];
            }

            net->deltas[l][i] = error * drelu(net->activations[l][i]);
        }
    }
}

void network_update_weights(Network* net, double learning_rate) {
    int l;

    if (net == NULL) {
        fprintf(stderr, "network_update_weights: net is NULL\n");
        return;
    }

    for (l = 0; l < net->n_layers - 1; l++) {
        int current_size = net->layer_sizes[l];
        int next_size = net->layer_sizes[l + 1];
        int j;

        for (j = 0; j < next_size; j++) {
            if (net->use_accumulated_gradients) {
                net->biases[l][j] -= learning_rate * net->bias_grads[l][j];

                for (int i = 0; i < current_size; i++) {
                    net->weights[l][i][j] -= learning_rate * net->weight_grads[l][i][j];
                }
            } else {
                net->biases[l][j] -= learning_rate * net->deltas[l + 1][j];

                for (int i = 0; i < current_size; i++) {
                    net->weights[l][i][j] -= learning_rate
                        * net->deltas[l + 1][j]
                        * net->activations[l][i];
                }
            }
        }
    }
}
