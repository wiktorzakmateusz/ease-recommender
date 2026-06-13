#ifndef EASE_GSL_MATH_H
#define EASE_GSL_MATH_H

// executing cholesky inversion in-place directly on the existing array to eliminate O(n^2) redundant
// memory allocation overhead, requiring double precision pointers to guarantee numerical stability 
// and prevent cancellation during the decomposition process
void invert_gram_matrix_gsl(double* G_data, int n_items);

#endif