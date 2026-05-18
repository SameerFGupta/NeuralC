#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "network.h"
#include "train.h"

int main(void) {
    const char* train_image_path = "train-images-idx3-ubyte";
    const char* train_label_path = "train-labels-idx1-ubyte";
    const char* test_image_path = "t10k-images-idx3-ubyte";
    const char* test_label_path = "t10k-labels-idx1-ubyte";
    int layer_sizes[] = {784, 128, 64, 10};
    Network* net;
    double* train_images;
    int* train_labels;
    double* test_images;
    int* test_labels;
    int train_count;
    int test_count;
    int train_label_count;
    int test_label_count;
    int train_rows;
    int train_cols;
    int test_rows;
    int test_cols;
    Dataset* train_set;
    Dataset* test_set;
    double final_accuracy;

    train_images = load_images(train_image_path, &train_count, &train_rows, &train_cols);
    if (train_images == NULL) {
        return 1;
    }

    train_labels = load_labels(train_label_path, &train_label_count);
    if (train_labels == NULL) {
        free(train_images);
        return 1;
    }

    test_images = load_images(test_image_path, &test_count, &test_rows, &test_cols);
    if (test_images == NULL) {
        free(train_images);
        free(train_labels);
        return 1;
    }

    test_labels = load_labels(test_label_path, &test_label_count);
    if (test_labels == NULL) {
        free(train_images);
        free(train_labels);
        free(test_images);
        return 1;
    }

    if (train_count != train_label_count || test_count != test_label_count ||
        train_rows * train_cols != 784 || test_rows * test_cols != 784) {
        fprintf(stderr, "Failed to load matching MNIST datasets\n");
        free(train_images);
        free(train_labels);
        free(test_images);
        free(test_labels);
        return 1;
    }

    train_set = make_dataset(train_images, train_labels, train_count, train_rows * train_cols, 10);
    test_set = make_dataset(test_images, test_labels, test_count, test_rows * test_cols, 10);
    if (train_set == NULL || test_set == NULL) {
        free_dataset(train_set);
        free_dataset(test_set);
        return 1;
    }

    net = network_create(layer_sizes, 4);
    if (net == NULL) {
        free_dataset(train_set);
        free_dataset(test_set);
        return 1;
    }

    network_init_xavier(net);
    train(net, train_set, test_set, 40, 32, 0.01);

    final_accuracy = evaluate(net, test_set);
    printf("Final Test Accuracy: %.2f%%\n", final_accuracy * 100.0);

    network_free(net);
    free_dataset(train_set);
    free_dataset(test_set);
    return 0;
}
