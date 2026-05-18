#ifndef DATA_H
#define DATA_H

typedef struct {
    double* images;
    int*    labels;
    int     n_samples;
    int     n_features;
    int     n_classes;
} Dataset;

double* load_images(const char* path, int* out_count, int* out_rows, int* out_cols);
int* load_labels(const char* path, int* out_count);
Dataset* make_dataset(double* images, int* labels, int n_samples, int n_features, int n_classes);
void free_dataset(Dataset* d);

#endif
