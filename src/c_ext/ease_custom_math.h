#ifndef EASE_CUSTOM_MATH_H
#define EASE_CUSTOM_MATH_H

// computing the dense gram matrix from a CSR sparse matrix, using double precision to guarantee 
// numerical stability during the subsequent cholesky inversion phase
void compute_gram_matrix_csr(const double* data, const int* indices, const int* indptr, 
                             int n_users, int n_items, double reg_weight, double* out_G);

// computing top-k recommendations for a specific user, using single precision float to halve the
// memory footprint and maximize hardware cache efficiency during inference
void predict_top_k(const float* B, int n_items, const int* interacted, 
                   int n_interacted, int K, int* out_top_k_indices);

#endif