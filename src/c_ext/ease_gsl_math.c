#include "ease_gsl_math.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

void invert_gram_matrix_gsl(double* G_data, int n_items) {
    // wrapping the raw c array in a gsl matrix view to achieve zero-copy memory access and prevent 
    // duplicate heap allocations
    gsl_matrix_view G_view = gsl_matrix_view_array(G_data, n_items, n_items);

    // performing cholesky decomposition, exploiting the symmetric positive-definite property of the
    // gram matrix to halve algorithmic complexity compared to standard LU decomposition
    gsl_linalg_cholesky_decomp(&G_view.matrix);

    // executing cholesky inversion, mutating the decomposed matrix in-place directly inside the 
    // python buffer to minimize RAM overhead
    gsl_linalg_cholesky_invert(&G_view.matrix);
}