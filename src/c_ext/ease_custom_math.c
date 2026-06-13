#include "ease_custom_math.h"
#include <stdlib.h>
#include <string.h>
#define BLOCK_SIZE 512

void compute_gram_matrix_csr(const double* data, const int* indices, const int* indptr, 
                             int n_users, int n_items, double reg_weight, double* out_G) {
                             
    // initializing the output matrix to zero before accumulation
    memset(out_G, 0, sizeof(double) * n_items * n_items);

    // using dynamic scheduling to balance the load since the number of interactions per user varies
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < n_users; u++) {
        int start = indptr[u];
        int end = indptr[u + 1];
        
        // computing only the upper triangle of the symmetric matrix to halve computational cost
        for (int i = start; i < end; i++) {
            int item_i = indices[i];
            double val_i = data[i];
            
            for (int j = i; j < end; j++) {
                int item_j = indices[j];
                double val_j = data[j];

                // atomic prevents race conditions when multiple threads update the same 
                // matrix cell
                #pragma omp atomic
                out_G[item_i * n_items + item_j] += val_i * val_j;
            }
        }
    }

    // using static scheduling because the workload per row is predictable and equal
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_items; i++) {
        for (int j = i; j < n_items; j++) {
            if (i == j) {
                out_G[i * n_items + j] += reg_weight;
            } else {
                // mirroring the computed upper triangle to the lower half instead of recalculating
                out_G[j * n_items + i] = out_G[i * n_items + j];
            }
        }
    }
}

void predict_top_k(const float* B, int n_items, const int* interacted, 
                   int n_interacted, int K, int* out_top_k_indices) {
    
    // O(1) lookup array to quickly filter out items the user has already interacted with
    char* is_interacted = (char*)calloc(n_items, sizeof(char));
    for(int i = 0; i < n_interacted; i++) {
        is_interacted[interacted[i]] = 1;
    }

    // maintaining a small array for top-k scores to avoid allocating an O(n_items) array per 
    // user
    float* top_scores = (float*)malloc(K * sizeof(float));
    for(int i = 0; i < K; i++) {
        top_scores[i] = -1e9f; 
        out_top_k_indices[i] = -1;
    }

    // processing items in fixed blocks to ensure the active memory fits entirely within the 
    // CPU L1 cache
    for (int b_start = 0; b_start < n_items; b_start += BLOCK_SIZE) {
        int b_end = (b_start + BLOCK_SIZE < n_items) ? b_start + BLOCK_SIZE : n_items;
        int current_block_size = b_end - b_start;

        // stack-allocating the block buffer to avoid the overhead of dynamic heap allocation
        float local_scores[BLOCK_SIZE] = {0.0f};

        // iterating users outside items to create a sequential memory access pattern for matrix B
        for (int i = 0; i < n_interacted; i++) {
            int row_offset = interacted[i] * n_items + b_start;
            
            for (int j = 0; j < current_block_size; j++) {
                int global_j = b_start + j;
                if (!is_interacted[global_j]) {
                    local_scores[j] += B[row_offset + j];
                }
            }
        }

        // inserting the block's scores into the top-k tracker immediately while the block data 
        // remains in cache
        for (int j = 0; j < current_block_size; j++) {
            int global_j = b_start + j;
            if (is_interacted[global_j]) continue;

            float score = local_scores[j];
            if (score > top_scores[K - 1]) {
                int pos = K - 1;
                // linear insertion sort is highly efficient since k is very small
                while (pos > 0 && score > top_scores[pos - 1]) {
                    top_scores[pos] = top_scores[pos - 1];
                    out_top_k_indices[pos] = out_top_k_indices[pos - 1];
                    pos--;
                }
                top_scores[pos] = score;
                out_top_k_indices[pos] = global_j;
            }
        }
    }

    free(is_interacted);
    free(top_scores);
}