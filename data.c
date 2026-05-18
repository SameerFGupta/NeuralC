#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"

static int read_be_int(FILE* fp) {
    unsigned char bytes[4];

    if (fread(bytes, 1, 4, fp) != 4) {
        return -1;
    }

    return ((int) bytes[0] << 24) |
           ((int) bytes[1] << 16) |
           ((int) bytes[2] << 8)  |
           (int) bytes[3];
}

double* load_images(const char* path, int* out_count, int* out_rows, int* out_cols) {
    FILE* fp;
    int magic;
    int num_images;
    int rows;
    int cols;
    int total_pixels;
    unsigned char* raw_pixels;
    double* images;
    int i;

    if (out_count == NULL || out_rows == NULL || out_cols == NULL) {
        fprintf(stderr, "load_images: output parameter is NULL\n");
        return NULL;
    }

    *out_count = 0;
    *out_rows = 0;
    *out_cols = 0;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open image file: %s\n", path);
        return NULL;
    }

    magic = read_be_int(fp);
    num_images = read_be_int(fp);
    rows = read_be_int(fp);
    cols = read_be_int(fp);

    if (magic != 2051) {
        fprintf(stderr, "Invalid image file magic number in %s: %d\n", path, magic);
        fclose(fp);
        return NULL;
    }

    if (num_images <= 0 || rows <= 0 || cols <= 0) {
        fprintf(stderr, "Invalid image dimensions in %s\n", path);
        fclose(fp);
        return NULL;
    }

    if (rows > 2147483647 / cols || num_images > 2147483647 / (rows * cols)) {
        fprintf(stderr, "Image file is too large: %s\n", path);
        fclose(fp);
        return NULL;
    }

    total_pixels = num_images * rows * cols;

    raw_pixels = (unsigned char*) malloc((size_t) total_pixels * sizeof(unsigned char));
    if (raw_pixels == NULL) {
        fprintf(stderr, "Failed to allocate raw image buffer\n");
        fclose(fp);
        return NULL;
    }

    if (fread(raw_pixels, sizeof(unsigned char), (size_t) total_pixels, fp) != (size_t) total_pixels) {
        fprintf(stderr, "Failed to read image data from %s\n", path);
        free(raw_pixels);
        fclose(fp);
        return NULL;
    }

    images = (double*) malloc((size_t) total_pixels * sizeof(double));
    if (images == NULL) {
        fprintf(stderr, "Failed to allocate normalized image buffer\n");
        free(raw_pixels);
        fclose(fp);
        return NULL;
    }

    for (i = 0; i < total_pixels; i++) {
        images[i] = (double) raw_pixels[i] / 255.0;
    }

    free(raw_pixels);
    fclose(fp);

    *out_count = num_images;
    *out_rows = rows;
    *out_cols = cols;

    return images;
}

int* load_labels(const char* path, int* out_count) {
    FILE* fp;
    int magic;
    int num_labels;
    unsigned char* raw_labels;
    int* labels;
    int i;

    if (out_count == NULL) {
        fprintf(stderr, "load_labels: output parameter is NULL\n");
        return NULL;
    }

    *out_count = 0;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open label file: %s\n", path);
        return NULL;
    }

    magic = read_be_int(fp);
    num_labels = read_be_int(fp);

    if (magic != 2049) {
        fprintf(stderr, "Invalid label file magic number in %s: %d\n", path, magic);
        fclose(fp);
        return NULL;
    }

    if (num_labels <= 0) {
        fprintf(stderr, "Invalid label count in %s\n", path);
        fclose(fp);
        return NULL;
    }

    raw_labels = (unsigned char*) malloc((size_t) num_labels * sizeof(unsigned char));
    if (raw_labels == NULL) {
        fprintf(stderr, "Failed to allocate raw label buffer\n");
        fclose(fp);
        return NULL;
    }

    if (fread(raw_labels, sizeof(unsigned char), (size_t) num_labels, fp) != (size_t) num_labels) {
        fprintf(stderr, "Failed to read labels from %s\n", path);
        free(raw_labels);
        fclose(fp);
        return NULL;
    }

    labels = (int*) malloc((size_t) num_labels * sizeof(int));
    if (labels == NULL) {
        fprintf(stderr, "Failed to allocate label array\n");
        free(raw_labels);
        fclose(fp);
        return NULL;
    }

    for (i = 0; i < num_labels; i++) {
        labels[i] = (int) raw_labels[i];
    }

    free(raw_labels);
    fclose(fp);

    *out_count = num_labels;

    return labels;
}

Dataset* make_dataset(double* images, int* labels, int n_samples, int n_features, int n_classes) {
    Dataset* dataset;

    dataset = (Dataset*) malloc(sizeof(Dataset));
    if (dataset == NULL) {
        fprintf(stderr, "Failed to allocate Dataset\n");
        return NULL;
    }

    dataset->images = images;
    dataset->labels = labels;
    dataset->n_samples = n_samples;
    dataset->n_features = n_features;
    dataset->n_classes = n_classes;

    return dataset;
}

void free_dataset(Dataset* d) {
    if (d == NULL) {
        return;
    }

    free(d->images);
    free(d->labels);
    free(d);
}
