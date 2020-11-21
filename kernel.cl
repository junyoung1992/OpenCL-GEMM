__kernel void vec_mul_1(const __global float *A,
                        const __global float *B,
                        __global float *C,
                        const int ROW_A, const int COL_A, const int COL_B)
{
    const int j = get_global_id(0);
    const int i = get_global_id(1);

    if (i >= COL_B || j >= ROW_A) return;

    for (int k = 0; k < COL_A; k++)
    {
        C[j * COL_B + i] += A[j * COL_A + k] * B[k * COL_B + i];
    }
}

__kernel void vec_mul_2(const __global float *A,
                        const __global float *B,
                        __global float *C,
                        const int ROW_A, const int COL_A, const int COL_B)
{
    const int j = get_global_id(0);
    const int i = get_global_id(1);

    if (i >= COL_B || j >= ROW_A) return;

    float sum = 0.0f;
    for (int k = 0; k < COL_A; k++)
    {
        sum += A[j * COL_A + k] * B[k * COL_B + i];
    }
    C[j * COL_B + i] = sum;
}

__kernel void vec_mul_3(const __global float *A,
                        const __global float *B,
                        __global float *C,
                        const int ROW_A, const int COL_A, const int COL_B)
{
    const int j = get_local_id(0);
    const int i = get_local_id(1);

    const int gj = get_group_id(0) * TS + j;
    const int gi = get_group_id(1) * TS + i;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    
    float sum = 0.0f;

    for (int t = 0; t < COL_A; t += TS)
    {
        const int tj = t + j;
        const int ti = t + i;

        if (gj < ROW_A && ti < COL_B)
        {
            Asub[j][i] = A[COL_A * gj + ti];
        }
        else
        {
            Asub[j][i] = 0.0f;
        }

        if (gi < COL_B && tj < ROW_A)
        {
            Bsub[j][i] = B[COL_B * tj + gi];
        }
        else
        {
            Bsub[j][i] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k < TS; k++)
        {
            sum += Asub[j][k] * Bsub[k][i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gi < COL_B && gj < ROW_A)
    {
        C[COL_B * gj + gi] = sum;
    }
}

__kernel void vec_mul_4(const __global float *A,
                        const __global float *B,
                        __global float *C,
                        const int ROW_A, const int COL_A, const int COL_B)
{
    const int j = get_local_id(0);
    const int i = get_local_id(1);

    const int gj = get_group_id(0) * TS + j;
    const int gi = get_group_id(1) * TS + i;

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    
    const int RTS = TS / WPT;
    float sum[WPT] = {0.0f};

    for (int t = 0; t < COL_A; t += TS)
    {
        for (int w = 0; w < WPT; w++)
        {
            const int tj = t + j;
            const int ti = t + i;

            if ((gj + (w * RTS)) < ROW_A && ti < COL_B)
            {
                Asub[j + (w * RTS)][i] = A[COL_A * (gj + (w * RTS)) + ti];
            }
            else
            {
                Asub[j + (w * RTS)][i] = 0.0f;
            }

            if (gi < COL_B && (tj + (w * RTS)) < ROW_A)
            {
                Bsub[j + (w * RTS)][i] = B[COL_B * (tj + (w * RTS)) + gi];
            }
            else
            {
                Bsub[j + (w * RTS)][i] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++)
        {
            for (int w = 0; w < WPT; w++)
            {
                sum[w] += Asub[j + (w * RTS)][k] * Bsub[k][i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WPT; w++)
    {
        if ((gj + (w * RTS)) < ROW_A && gi < COL_B)
        {
            C[COL_B * (gj  + (w * RTS)) + gi] = sum[w];
        }
    }
}
