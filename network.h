#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    int      n_layers;
    int*     layer_sizes;

    double*** weights;
    double**   biases;
    double*** weight_grads;
    double**   bias_grads;

    double**   activations;
    double**   deltas;
    int        use_accumulated_gradients;
} Network;

double** alloc_matrix(int rows, int cols);
void free_matrix(double** matrix, int rows);
Network* network_create(int* layer_sizes, int n_layers);
void network_init_xavier(Network* net);
void network_forward(Network* net, double* input);
double network_cross_entropy_loss(Network* net, int true_label);
void network_output_deltas(Network* net, int true_label);
void network_backward(Network* net);
void network_update_weights(Network* net, double learning_rate);
void network_free(Network* net);

#endif
