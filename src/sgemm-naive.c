#include <omp.h>

void sgemm( int m, int n, float *A, float *C )
{
#pragma omp parallel
    {
#pragma omp for schedule(dynamic) nowait
        for (int j = 0; j < m; j++) 
            for (int k = 0; k < n; k++)
                for (int i = 0; i < m; ++i) 
                    C[i+j*m] += A[i+k*m] * A[j+k*m];
    }
}
