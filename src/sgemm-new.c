#include <emmintrin.h> // header file for the SSE intrinsics we're going to use
#include <omp.h>

#define ADD _mm_add_ps
#define MUL _mm_mul_ps
#define STORE _mm_storeu_ps
#define LOADU _mm_loadu_ps
#define LOAD1 _mm_load1_ps
#define NUM_THREADS 4

void sgemm(int m, int n, float *A, float *C)
{
    const size_t    mSquared = m * m;
    const int       nb64 = (m / 64);
    float*          Abegin = A;

    //omp_set_num_threads(NUM_THREADS);
    // 64 by 64 part
#pragma omp parallel
    {
#pragma omp for schedule(dynamic) nowait
        for (size_t count = 0; count < nb64; ++count)
        {
            A = Abegin + count * 64;
            float* myC = C + count * 64;

            for (int k = 0; k < 64; ++k) 
            {
                __m128 tab[16];
                __m128 res;
                A = Abegin + count * 64 + k * m;

                // Loop Unrolling
                tab[0] = LOADU(A);
                tab[1] = LOADU(A + 4);
                tab[2] = LOADU(A + 8);
                tab[3] = LOADU(A + 12);
                tab[4] = LOADU(A + 16);
                tab[5] = LOADU(A + 20);
                tab[6] = LOADU(A + 24);
                tab[7] = LOADU(A + 28);
                tab[8] = LOADU(A + 32);
                tab[9] = LOADU(A + 36);
                tab[10] = LOADU(A + 40);
                tab[11] = LOADU(A + 44);
                tab[12] = LOADU(A + 48);
                tab[13] = LOADU(A + 52);
                tab[14] = LOADU(A + 56);
                tab[15] = LOADU(A + 60);

                size_t j = 0;

                for (size_t tmp = 0; tmp < mSquared; tmp += m)
                {
                    res = LOAD1(Abegin + k * m + j);

                    for (int i = 0; i < 16; ++i)
                        STORE(myC + tmp + (i * 4), (ADD(LOADU
                                        (myC + tmp + i * 4), (MUL(tab[i], res)))));
                    ++j;
                }
            }
        }
    }

    // Multiples of 4 under 64 here
    size_t count = nb64;
    C += count * 64;
    int leftOver = m - count * 64;
    int leftOverDivBy4 = leftOver / 4;
    int leftOverMod4 = leftOver % 4;

    if (count * 64 != m)
    {
        float* A2 = Abegin;
        A = Abegin + count * 64;
        for (int k = 0; k < 64; ++k, A += m) 
        {
            __m128          tab[16];
            __m128          res;
            for (unsigned short j = 0; j < leftOverDivBy4; ++j)
                tab[j] = LOADU(A + j * 4);

            for (size_t tmp = 0; tmp < mSquared; tmp += m, ++A2)
            {
                res = LOAD1(A2);

                for (int i = 0; i < leftOverDivBy4; ++i)
                    STORE(C + tmp + (i * 4), (ADD(LOADU
                                    (C + tmp + i * 4), (MUL(tab[i], res)))));
            }
        }
        C += leftOverDivBy4 * 4;
    }

    A = Abegin + count * 64 + leftOverDivBy4 * 4;

    // 1 2 or 3 leftovers
    if (leftOverMod4)
    {
        for (int k = 0; k < 64; ++k)
            for (int j = 0; j < leftOverMod4; ++j)
                for (int i = 0; i < m; ++i) 
                    C[j+i*m] += A[j+k*m] * Abegin[i+k*m];
    }
}
