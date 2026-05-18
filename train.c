#include <stdio.h>
#include <stdlib.h>

#include "train.h"

static void zero_accumulators(Network* net) {
    int l;

    for (l = 0; l < net->n_layers - 1; l++) {
        int rows = net->layer_sizes[l];
        int cols = net->layer_sizes[l + 1];
        int i;
        int j;

        for (j = 0; j < cols; j++) {
            net->bias_grads[l][j] = 0.0;
        }

        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                net->weight_grads[l][i][j] = 0.0;
            }
        }
    }
}

static void average_accumulators(Network* net, int batch_count) {
    int l;
    double scale = 1.0 / (double) batch_count;

    for (l = 0; l < net->n_layers - 1; l++) {
        int rows = net->layer_sizes[l];
        int cols = net->layer_sizes[l + 1];
        int i;
        int j;

        for (j = 0; j < cols; j++) {
            net->bias_grads[l][j] *= scale;
        }

        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                net->weight_grads[l][i][j] *= scale;
            }
        }
    }
}

static void accumulate_gradients(Network* net) {
    int l;

    for (l = 0; l < net->n_layers - 1; l++) {
        int input_size = net->layer_sizes[l];
        int output_size = net->layer_sizes[l + 1];
        int i;
        int j;

        for (j = 0; j < output_size; j++) {
            net->bias_grads[l][j] += net->deltas[l + 1][j];
        }

        for (i = 0; i < input_size; i++) {
            for (j = 0; j < output_size; j++) {
                net->weight_grads[l][i][j] +=
                    net->deltas[l + 1][j] * net->activations[l][i];
            }
        }
    }
}

static int argmax(double* values, int n) {
    int best_index = 0;
    int i;

    for (i = 1; i < n; i++) {
        if (values[i] > values[best_index]) {
            best_index = i;
        }
    }

    return best_index;
}

void shuffle(int* indices, int n) {
    int i;

    for (i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];

        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

double evaluate(Network* net, Dataset* test_set) {
    int correct = 0;
    int i;

    for (i = 0; i < test_set->n_samples; i++) {
        double* input = test_set->images + ((size_t) i * (size_t) test_set->n_features);
        int predicted_class;

        network_forward(net, input);
        predicted_class = argmax(
            net->activations[net->n_layers - 1],
            net->layer_sizes[net->n_layers - 1]
        );

        if (predicted_class == test_set->labels[i]) {
            correct++;
        }
    }

    return (double) correct / (double) test_set->n_samples;
}

void train(Network* net, Dataset* train_set, Dataset* test_set,
           int epochs, int batch_size, double learning_rate) {
    int* indices;
    int epoch;

    indices = (int*) malloc((size_t) train_set->n_samples * sizeof(int));
    if (indices == NULL) {
        fprintf(stderr, "train: failed to allocate shuffled indices\n");
        return;
    }

    for (epoch = 1; epoch <= epochs; epoch++) {
        int start;
        double epoch_loss = 0.0;

        for (int i = 0; i < train_set->n_samples; i++) {
            indices[i] = i;
        }
        shuffle(indices, train_set->n_samples);

        zero_accumulators(net);

        for (start = 0; start < train_set->n_samples; start += batch_size) {
            int end = start + batch_size;
            int batch_count;
            int b;

            if (end > train_set->n_samples) {
                end = train_set->n_samples;
            }

            batch_count = end - start;
            zero_accumulators(net);

            for (b = start; b < end; b++) {
                int idx = indices[b];
                double* input = train_set->images + ((size_t) idx * (size_t) train_set->n_features);

                network_forward(net, input);
                network_output_deltas(net, train_set->labels[idx]);
                network_backward(net);
                accumulate_gradients(net);
            }

            average_accumulators(net, batch_count);
            net->use_accumulated_gradients = 1;
            network_update_weights(net, learning_rate);
            net->use_accumulated_gradients = 0;
            zero_accumulators(net);
        }

        for (int i = 0; i < train_set->n_samples; i++) {
            double* input = train_set->images + ((size_t) i * (size_t) train_set->n_features);

            network_forward(net, input);
            epoch_loss += network_cross_entropy_loss(net, train_set->labels[i]);
        }

        epoch_loss /= (double) train_set->n_samples;

        printf(
            "Epoch %2d | Loss: %.4f | Test Accuracy: %.2f%%\n",
            epoch,
            epoch_loss,
            evaluate(net, test_set) * 100.0
        );
    }

    free(indices);
}
