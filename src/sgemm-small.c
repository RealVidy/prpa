#include <emmintrin.h> /* header file for the SSE intrinsics we gonna use */

#define ADD _mm_add_ps
#define MUL _mm_mul_ps
#define STORE _mm_storeu_ps
#define LOADU _mm_loadu_ps
#define LOAD1 _mm_load1_ps

void sgemm64First(int m, int n, float *A, float *C)
{
    __m128 tab[16]; 
    __m128 res;
    int tmp, tmp2;
    int k = 0, j;

    do 
    {
        tmp2 = k * m;
        tab[0] = LOADU(&A[tmp2]);
        tab[1] = LOADU(&A[4 + tmp2]);
        tab[2] = LOADU(&A[8 + tmp2]);
        tab[3] = LOADU(&A[12 + tmp2]);
        tab[4] = LOADU(&A[16 + tmp2]);
        tab[5] = LOADU(&A[20 + tmp2]);
        tab[6] = LOADU(&A[24 + tmp2]);
        tab[7] = LOADU(&A[28 + tmp2]);
        tab[8] = LOADU(&A[32 + tmp2]);
        tab[9] = LOADU(&A[36 + tmp2]);
        tab[10] = LOADU(&A[40 + tmp2]);
        tab[11] = LOADU(&A[44 + tmp2]);
        tab[12] = LOADU(&A[48 + tmp2]);
        tab[13] = LOADU(&A[52 + tmp2]);
        tab[14] = LOADU(&A[56 + tmp2]);
        tab[15] = LOADU(&A[60 + tmp2]);

        j = 0;
        do
        {
            res = LOAD1(&A[j + tmp2]);
            tmp = j * m;
            STORE(&C[tmp], (ADD(LOADU
                            (&C[tmp]), (MUL(tab[0], res)))));
            STORE(&C[4 + tmp], (ADD(LOADU
                            (&C[4 + tmp]), (MUL(tab[1], res)))));
            STORE(&C[8 + tmp], (ADD(LOADU
                            (&C[8 + tmp]), (MUL(tab[2], res)))));
            STORE(&C[12 + tmp], (ADD(LOADU
                            (&C[12 + tmp]), (MUL(tab[3], res)))));
            STORE(&C[16 + tmp], (ADD(LOADU
                            (&C[16 + tmp]), (MUL(tab[4], res)))));
            STORE(&C[20 + tmp], (ADD(LOADU
                            (&C[20 + tmp]), (MUL(tab[5], res)))));
            STORE(&C[24 + tmp], (ADD(LOADU
                            (&C[24 + tmp]), (MUL(tab[6], res)))));
            STORE(&C[28 + tmp], (ADD(LOADU
                            (&C[28 + tmp]), (MUL(tab[7], res)))));
            STORE(&C[32 + tmp], (ADD(LOADU
                            (&C[32 + tmp]), (MUL(tab[8], res)))));
            STORE(&C[36 + tmp], (ADD(LOADU
                            (&C[36 + tmp]), (MUL(tab[9], res)))));
            STORE(&C[40 + tmp], (ADD(LOADU
                            (&C[40 + tmp]), (MUL(tab[10], res)))));
            STORE(&C[44 + tmp], (ADD(LOADU
                            (&C[44 + tmp]), (MUL(tab[11], res)))));
            STORE(&C[48 + tmp], (ADD(LOADU
                            (&C[48 + tmp]), (MUL(tab[12], res)))));
            STORE(&C[52 + tmp], (ADD(LOADU
                            (&C[52 + tmp]), (MUL(tab[13], res)))));
            STORE(&C[56 + tmp], (ADD(LOADU
                            (&C[56 + tmp]), (MUL(tab[14], res)))));
            STORE(&C[60 + tmp], (ADD(LOADU
                            (&C[60 + tmp]), (MUL(tab[15], res)))));
            j++;
        } while (j < m);
        k++;
    } while (k < 64);
}


void sgemm64(int m, int n, float *A, float *Abegin, float *C)
{
    __m128 tab[16];
    __m128 res;
    int tmp;

    for (int k = 0; k < 64; k++) 
    {
        for (int x = 0; x < 16; x++)
            tab[x] = LOADU(&A[(x * 4) + (k * m)]);

        for (int j = 0; j < m; j++)
        {
            res = LOAD1(&Abegin[j + (k * m)]);

            tmp = j * m;

            for (int i = 0; i < 16; i++)
            {
                STORE(&C[tmp + (i * 4)], 
                        (ADD(LOADU(&C[tmp + (i * 4)]),
                             (MUL(tab[i], res)))));
            }
        }
    }
}

void sgemm32(int m, int n, float *A, float *Abegin, float *C)
{
    __m128 tab[8]; // using 16 different variables would be a bit faster.
    __m128 res;
    int tmp;

    for (int k = 0; k < 64; k++) 
    {
        for (int x = 0; x < 8; x++)
            tab[x] = LOADU(&A[(x * 4) + (k * m)]);

        for (int j = 0; j < m; j++)
        {
            res = LOAD1(&Abegin[j + (k * m)]);

            tmp = j * m;

            for (int i = 0; i < 8; i++)
            {
                STORE(&C[tmp + (i * 4)], 
                        (ADD(LOADU(&C[tmp + (i * 4)]),
                             (MUL(tab[i], res)))));
            }
        }
    }
}

void sgemm16(int m, int n, float *A, float *Abegin, float *C)
{
    __m128 tab[4]; // using 16 different variables would be a bit faster.
    __m128 res;
    int tmp;

    for (int k = 0; k < 64; k++) 
    {
        for (int x = 0; x < 4; x++)
            tab[x] = LOADU(&A[(x * 4) + (k * m)]);

        for (int j = 0; j < m; j++)
        {
            res = LOAD1(&Abegin[j + (k * m)]);

            tmp = j * m;

            for (int i = 0; i < 4; i++)
            {
                STORE(&C[tmp + (i * 4)], 
                        (ADD(LOADU(&C[tmp + (i * 4)]),
                             (MUL(tab[i], res)))));
            }
        }
    }
}

void sgemm8(int m, int n, float *A, float *Abegin, float *C)
{
    __m128 tab[2]; // using 16 different variables would be a bit faster.
    __m128 res;
    int tmp;

    for (int k = 0; k < 64; k++) 
    {
        for (int x = 0; x < 2; x++)
            tab[x] = LOADU(&A[(x * 4) + (k * m)]);

        for (int j = 0; j < m; j++)
        {
            res = LOAD1(&Abegin[j + (k * m)]);

            tmp = j * m;

            for (int i = 0; i < 2; i++)
            {
                STORE(&C[tmp + (i * 4)], 
                        (ADD(LOADU(&C[tmp + (i * 4)]),
                             (MUL(tab[i], res)))));
            }
        }
    }
}

void sgemm4(int m, int n, float *A, float *Abegin, float *C)
{
    __m128 tab; // using 16 different variables would be a bit faster.
    __m128 res;
    int tmp;

    for (int k = 0; k < 64; k++) 
    {
        tab = LOADU(&A[(k * m)]);

        for (int j = 0; j < m; j++)
        {
            res = LOAD1(&Abegin[j + (k * m)]);

            tmp = j * m;
            STORE(&C[tmp], 
                    (ADD(LOADU(&C[tmp]),
                         (MUL(tab, res)))));
        }
    }
}

void sgemm0(int offsetY, int m, int n, float *A, float *C)
{
    __m128 tab; // using 16 different variables would be a bit faster.
    __m128 res;
    int tmp;

    for (int k = 0; k < 64; k++) 
    {
        tab = LOADU(&A[offsetY + (k * m)]);

        for (int j = 0; j < m; j++)
        {
            for (int l = 0; l < (m - offsetY); l++)
            {
                C[offsetY + l + j * m] += 
                    A[j + (k * m)] * A[offsetY + l + (k * m)];
            }
        }
    }
}

void sgemm(int m, int n, float *A, float *C)
{
    int offsetY;

    for (offsetY = 0; m - offsetY > 63; offsetY += 64)
    {
        if (offsetY == 0)
            sgemm64First(m, n, A, C);
        else
            sgemm64(m, n, A + offsetY, A, C + offsetY);
    }

    if (m - offsetY >= 32)
    {
        sgemm32(m, n, A + offsetY, A, C + offsetY);
        offsetY += 32;
    }

    if (m - offsetY >= 16)
    {
        sgemm16(m, n, A + offsetY, A, C + offsetY);
        offsetY += 16;
    }

    if (m - offsetY >= 8)
    {
        sgemm8(m, n, A + offsetY, A, C + offsetY);
        offsetY += 8;
    }

    if (m - offsetY >= 4)
    {
        sgemm4(m, n, A + offsetY, A, C + offsetY);
        offsetY += 4;
    }

    if (m - offsetY > 0)
    {
        sgemm0(offsetY, m, n, A, C);
    }
    /*__m128 tab[16]; // using 16 different variables would be a bit faster.
      __m128 res;
      int i, left;

      for (int k = 0; k < n; k++) 
      {
      for (int z = 0; (m - z) > 0; z += 64) // mulz + 0.2 Gflops 
      {
      for (i = 0; (m - ((i + 1) * 4 + z)) >= 0 && i < 16; i++)
      tab[i] = LOADU(&A[(i * 4) + z + (k * m)]);
      left = m - (z + i * 4);

      for (int j = 0; j < m; j++)
      {
      res = LOAD1(&A[j + (k * m)]);

      for (i = 0; i < 16 && (m - ((i + 1) * 4 + z)) >= 0; i++)
      {
      STORE(&C[(i * 4) + z + j * m], 
      (ADD(LOADU(&C[(i * 4) + z + j * m]),
      (MUL(tab[i], res)))));
      }

      for (int l = 0; left < 4 && l < left; l++)
      {
      C[(i * 4) + z + l + j * m] += 
      A[j + (k * m)] * A[(i * 4) + z + l + (k * m)];
      }
      }
      }
      }
      */
}
