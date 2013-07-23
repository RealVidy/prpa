#include <emmintrin.h> // header file for the SSE size_trinsics we're going to use
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <google/profiler.h>

#define ADD _mm_add_ps
#define MUL _mm_mul_ps
#define STORE _mm_store_ps
#define LOAD _mm_load_ps
#define LOADU _mm_loadu_ps
#define LOAD1 _mm_load1_ps

#define min(a, b) (a < b ? a : b)
#define BLOCK_SIZE 64

void cpyAndResetBuf(float *src, float *dst, const size_t m,
        const size_t colLength, const size_t lineLength)
{
    const size_t borderOfC = min(BLOCK_SIZE, colLength);

    // Copy to destination
    for (size_t j = 0; j < borderOfC; j++)
        memcpy(dst + j * m, src + j * BLOCK_SIZE,
                (!lineLength ? BLOCK_SIZE : lineLength) * 4);

    // Reset buffer
    memset(src, 0, BLOCK_SIZE * BLOCK_SIZE * 4);
}


void sgemm(size_t m, size_t n, float *A, float *C)
{
    // borderBuf will avoid segfaults.
    float borderBuf[BLOCK_SIZE] = {0};
    const size_t border = (m - (m - (m / BLOCK_SIZE) * BLOCK_SIZE));
    const size_t nb64 = min(border + 1, m);

    // get the border buffer. Most of the time it will be empty at the end.
    memcpy(borderBuf, (A + (n - 1) * m + border), (4 * (m - border)));

    omp_set_num_threads(min(8, border / 64));

    //ProfilerStart("mybin.prof");
#pragma omp parallel for schedule(dynamic)
    for (size_t k = 0; k < nb64; k += 64)
    {
        const float *a = A + k;
        float *c = C + k;
        const size_t borderOfA = ((k == border) ? 1 : 0);
        float cBuf[BLOCK_SIZE * BLOCK_SIZE] = {0};

        for (size_t cIter = 0; cIter < border + 1; cIter += BLOCK_SIZE)
        {
            for (size_t i = 0; i < n; i++)
            {
                __m128  tab0, tab1, tab2, tab3;
                __m128  tab4, tab5, tab6, tab7;
                __m128  tab8, tab9, tab10, tab11;
                __m128  tab12, tab13, tab14, tab15;
                const float* a2 = a + i * m;

                // Are we at the very end of A? Avoid SEGFAULT
                if (borderOfA && (i == n - 1))
                {
                    /* Load from border buffer */
                    tab0 = LOAD(borderBuf);
                    tab1 = LOAD(borderBuf + 4);
                    tab2 = LOAD(borderBuf + 8);
                    tab3 = LOAD(borderBuf + 12);
                    tab4 = LOAD(borderBuf + 16);
                    tab5 = LOAD(borderBuf + 20);
                    tab6 = LOAD(borderBuf + 24);
                    tab7 = LOAD(borderBuf + 28);
                    tab8 = LOAD(borderBuf + 32);
                    tab9 = LOAD(borderBuf + 36);
                    tab10 = LOAD(borderBuf + 40);
                    tab11 = LOAD(borderBuf + 44);
                    tab12 = LOAD(borderBuf + 48);
                    tab13 = LOAD(borderBuf + 52);
                    tab14 = LOAD(borderBuf + 56);
                    tab15 = LOAD(borderBuf + 60);
                }
                else
                {
                    tab0 = LOADU(a2);
                    tab1 = LOADU(a2 + 4);
                    tab2 = LOADU(a2 + 8);
                    tab3 = LOADU(a2 + 12);
                    tab4 = LOADU(a2 + 16);
                    tab5 = LOADU(a2 + 20);
                    tab6 = LOADU(a2 + 24);
                    tab7 = LOADU(a2 + 28);
                    tab8 = LOADU(a2 + 32);
                    tab9 = LOADU(a2 + 36);
                    tab10 = LOADU(a2 + 40);
                    tab11 = LOADU(a2 + 44);
                    tab12 = LOADU(a2 + 48);
                    tab13 = LOADU(a2 + 52);
                    tab14 = LOADU(a2 + 56);
                    tab15 = LOADU(a2 + 60);
                }

                const float *a3 = A + i * m + cIter;
                const size_t borderOfC = min(BLOCK_SIZE, m - cIter);

                // Stops when we are at the end of A (with line length < 64)
                for (size_t j = 0; j < borderOfC; j++)
                {
                    const size_t dst = j * 64;
                    __m128 res = LOAD1(a3 + j);

                    STORE(cBuf + dst, ADD(LOAD(cBuf + dst), (MUL(tab0, res))));
                    STORE(cBuf + dst + 4, ADD(LOAD(cBuf + dst + 4), (MUL(tab1, res))));
                    STORE(cBuf + dst + 8, ADD(LOAD(cBuf + dst + 8), (MUL(tab2, res))));
                    STORE(cBuf + dst + 12, ADD(LOAD(cBuf + dst + 12), (MUL(tab3, res))));
                    STORE(cBuf + dst + 16, ADD(LOAD(cBuf + dst + 16), (MUL(tab4, res))));
                    STORE(cBuf + dst + 20, ADD(LOAD(cBuf + dst + 20), (MUL(tab5, res))));
                    STORE(cBuf + dst + 24, ADD(LOAD(cBuf + dst + 24), (MUL(tab6, res))));
                    STORE(cBuf + dst + 28, ADD(LOAD(cBuf + dst + 28), (MUL(tab7, res))));
                    STORE(cBuf + dst + 32, ADD(LOAD(cBuf + dst + 32), (MUL(tab8, res))));
                    STORE(cBuf + dst + 36, ADD(LOAD(cBuf + dst + 36), (MUL(tab9, res))));
                    STORE(cBuf + dst + 40, ADD(LOAD(cBuf + dst + 40), (MUL(tab10, res))));
                    STORE(cBuf + dst + 44, ADD(LOAD(cBuf + dst + 44), (MUL(tab11, res))));
                    STORE(cBuf + dst + 48, ADD(LOAD(cBuf + dst + 48), (MUL(tab12, res))));
                    STORE(cBuf + dst + 52, ADD(LOAD(cBuf + dst + 52), (MUL(tab13, res))));
                    STORE(cBuf + dst + 56, ADD(LOAD(cBuf + dst + 56), (MUL(tab14, res))));
                    STORE(cBuf + dst + 60, ADD(LOAD(cBuf + dst + 60), (MUL(tab15, res))));
                }
            }

            /* Copy results and reset buffer */
            // far Bottom right of C
            if (k == border && cIter == border)
                cpyAndResetBuf(cBuf, c + cIter * m, m, m - border, m - border);
            // Far right of C
            else if (cIter == border)
                cpyAndResetBuf(cBuf, c + cIter * m, m, m - border, 0);
            // Far bottom of C
            else if (k == border)
                cpyAndResetBuf(cBuf, c + cIter * m, m, BLOCK_SIZE, m - border);
            // Normal 64 * 64 block to be copied
            else
                cpyAndResetBuf(cBuf, c + cIter * m, m, BLOCK_SIZE, 0);
        }
    }
    //ProfilerStop();
}
