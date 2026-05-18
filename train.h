#ifndef TRAIN_H
#define TRAIN_H

#include "data.h"
#include "network.h"

void shuffle(int* indices, int n);
void train(Network* net, Dataset* train_set, Dataset* test_set,
           int epochs, int batch_size, double learning_rate);
double evaluate(Network* net, Dataset* test_set);

#endif
