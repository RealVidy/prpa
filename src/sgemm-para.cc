#include <emmintrin.h> // header file for the SSE intrinsics we're gonna use
#include <omp.h>

#define ADD _mm_add_ps
#define MUL _mm_mul_ps
#define STORE _mm_storeu_ps
#define LOADU _mm_loadu_ps
#define LOAD1 _mm_load1_ps

size_t mSquared = 0;
float* Abegin = NULL;

void sgemm64First(int m, int n, float* A, float* C)
{
    __m128 tab[16]; 
    __m128 res;
    float* A2 = A;

    for (int k = 0; k < 64; ++k)
    {
        // Some loop Unrolling
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

        for (size_t tmp = 0; tmp < mSquared; tmp += m, ++A2)
        {
            res = LOAD1(A2);

            for (int i = 0; i < 16; ++i)
                STORE(C + tmp + (i * 4), (ADD(LOADU
                                (C + tmp + i * 4), (MUL(tab[i], res)))));
        }
        A += m;
    }
}

void sgemm64(int m, int n, float *A, float *C)
{
    __m128 tab[16];
    __m128 res;
    float* A2 = Abegin;

    for (int k = 0; k < 64; ++k, A += m) 
    {
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

        for (size_t tmp = 0; tmp < mSquared; tmp += m, ++A2)
        {
            res = LOAD1(A2);

            for (int i = 0; i < 16; ++i)
                STORE(C + tmp + (i * 4), (ADD(LOADU
                                (C + tmp + i * 4), (MUL(tab[i], res)))));
        }
    }
}

void sgemm32(int m, int n, float *A, float *C)
{
    __m128 tab[8]; // using 16 different variables would be a bit faster.
    __m128 res;
    float* A2 = Abegin;

    for (int k = 0; k < 64; k++, A += m) 
    {
        for (int x = 0; x < 8; ++x)
            tab[x] = LOADU(A + (x * 4));

        for (size_t tmp = 0; tmp < mSquared; tmp += m, ++A2)
        {
            res = LOAD1(A2);

            for (int i = 0; i < 8; i += 4)
            {
                STORE(C + tmp + (i * 4), (ADD(LOADU(C + tmp + (i * 4)),
                                (MUL(tab[i], res)))));
                STORE(C + tmp + ((i + 1) * 4), (ADD(LOADU(C + tmp + ((i + 1) * 4)),
                                (MUL(tab[(i + 1)], res)))));
                STORE(C + tmp + ((i + 2) * 4), (ADD(LOADU(C + tmp + ((i + 2) * 4)),
                                (MUL(tab[(i + 2)], res)))));
                STORE(C + tmp + ((i + 3) * 4), (ADD(LOADU(C + tmp + ((i + 3) * 4)),
                                (MUL(tab[(i + 3)], res)))));
            } 
        }
    }
}

void sgemm16(int m, int n, float *A, float *C)
{
    __m128 tab[4]; // using 16 different variables would be a bit faster.
    __m128 res;
    int tmp = 0;

    for (int k = 0; k < 64; k++) 
    {
        for (int x = 0; x < 4; x += 4)
        {
            tab[x] = LOADU(&A[(x * 4) + (k * m)]);
            tab[x + 1] = LOADU(&A[((x + 1) * 4) + (k * m)]);
            tab[x + 2] = LOADU(&A[((x + 2) * 4) + (k * m)]);
            tab[x + 3] = LOADU(&A[((x + 3) * 4) + (k * m)]);
        }

        tmp = 0;
        for (int j = 0; j < m; ++j, tmp += m)
        {
            res = LOAD1(&Abegin[j + (k * m)]);

            STORE(&C[tmp], (ADD(LOADU(&C[tmp]),
                            (MUL(tab[0], res)))));
            STORE(&C[tmp + 4], (ADD(LOADU(&C[tmp + 4]),
                            (MUL(tab[1], res)))));
            STORE(&C[tmp + 8], (ADD(LOADU(&C[tmp + 8]),
                            (MUL(tab[2], res)))));
            STORE(&C[tmp + 12], (ADD(LOADU(&C[tmp + 12]),
                            (MUL(tab[3], res)))));
        }
    }
}

void sgemm8(int m, int n, float *A, float *C)
{
    __m128 tab[2]; // using 16 different variables would be a bit faster.
    __m128 res;
    int tmp = 0;

    for (int k = 0; k < 64; k++) 
    {
        tab[0] = LOADU(&A[k * m]);
        tab[1] = LOADU(&A[4 + (k * m)]);

        tmp = 0;
        for (int j = 0; j < m; ++j, tmp += m)
        {
            res = LOAD1(&Abegin[j + (k * m)]);

            STORE(&C[tmp], (ADD(LOADU(&C[tmp]), (MUL(tab[0], res)))));
            STORE(&C[tmp + 4], (ADD(LOADU(&C[tmp + 4]), (MUL(tab[1], res)))));
        }
    }
}

void sgemm4(int m, int n, float *A, float *C)
{
    __m128 tab;
    __m128 res;
    int tmp = 0;

    for (int k = 0; k < 64; k++) 
    {
        tab = LOADU(&A[(k * m)]);

        tmp = 0;
        for (int j = 0; j < m; ++j, tmp += m)
        {
            res = LOAD1(&Abegin[j + (k * m)]);

            STORE(&C[tmp], (ADD(LOADU(&C[tmp]), (MUL(tab, res)))));
        }
    }
}

void sgemm0(int offsetY, int m, int n, float* A, float *C)
{
    for (int k = 0; k < 64; k++) 
        for (int j = 0; j < m; ++j)
            for (int l = 0; l < (m - offsetY); l++)
                C[offsetY + l + j * m] +=
                    A[j + (k * m)] * A[offsetY + l + (k * m)];
}

void sgemm(int m, int n, float *A, float *C)
{
    int offsetY = 64;
    //int offsetY = 0;
    mSquared = m * m;
    const int nb64 = (m / 64);
    Abegin = A;

    // Somehow having a special function for first 64 gives better perfs
    if (m > 63)
        sgemm64First(m, n, A, C);

    for (size_t i = 0; i < nb64 - 1; ++i, offsetY += 64)
        sgemm64(m, n, A + offsetY, C + offsetY);
    /*
       parallel_for(blocked_range<size_t>(0, nb64 - 1, 1),
       [=, &offsetY](const blocked_range<size_t>&r) -> void
       {
       for (size_t i = r.begin(); i != r.end(); ++i, offsetY += 64)
       sgemm64(m, n, A + offsetY, C + offsetY);
       });
       */
    //MatrixComputer(m, n, A + offsetY, C + offsetY));

    if (m - offsetY >= 32)
    {
        sgemm32(m, n, A + offsetY, C + offsetY);
        offsetY += 32;
    }

    if (m - offsetY >= 16)
    {
        sgemm16(m, n, A + offsetY, C + offsetY);
        offsetY += 16;
    }

    if (m - offsetY >= 8)
    {
        sgemm8(m, n, A + offsetY, C + offsetY);
        offsetY += 8;
    }

    if (m - offsetY >= 4)
    {
        sgemm4(m, n, A + offsetY, C + offsetY);
        offsetY += 4;
    }

    if (m - offsetY > 0)
    {
        sgemm0(offsetY, m, n, A, C);
    }
}
