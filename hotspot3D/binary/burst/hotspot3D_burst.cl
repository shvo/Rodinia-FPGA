__kernel __attribute__ ((reqd_work_group_size(1,1,1)))
void hotspot3D_opt1(__global float *p, 
                    __global float *tIn, 
                    __global float *tOut,
                    float sdc,
                    int nx, int ny, int nz,
                    float ce, float cw, 
                    float cn, float cs,
                    float ct, float cb, 
                    float cc) 
{
    float amb_temp = 80.0;
    __local float p_linebuf0[512*8];
    __local float tIn_linebuf0[512*8], tIn_linebuf1[512*8], tIn_linebuf2[512*8];
    __local float tOut_linebuf0[512*8];
    
    for (int line = 0; line < 512; ++line) {
        
        // Fetch lines along the 0-dimension
        if (line == 0)  {
            for (int layer = 0; layer < 8; ++layer) {
                async_work_group_copy(&p_linebuf0[layer*512], &p[line*512 + layer*512*512], 512, 0);
                async_work_group_copy(&tIn_linebuf0[layer*512], &tIn[line*512 + layer*512*512], 512, 0);
                async_work_group_copy(&tIn_linebuf1[layer*512], &tIn[line*512 + layer*512*512], 512, 0);
                async_work_group_copy(&tIn_linebuf2[layer*512], &tIn[(line+1)*512 + layer*512*512], 512, 0);
            }
        }
        else if (line < 511) {
            if (line % 3 == 0) {
                for (int layer = 0; layer < 8; ++layer) {
                    async_work_group_copy(&p_linebuf0[layer*512], &p[line*512 + layer*512*512], 512, 0);
                    async_work_group_copy(&tIn_linebuf0[layer*512], &tIn[(line+1)*512 + layer*512*512], 512, 0);
                }
            }
            else if (line %3 == 1) {
                for (int layer = 0; layer < 8; ++layer) {
                    async_work_group_copy(&p_linebuf0[layer*512], &p[line*512 + layer*512*512], 512, 0);
                    async_work_group_copy(&tIn_linebuf1[layer*512], &tIn[(line+1)*512 + layer*512*512], 512, 0);
                }
            }
            else {
                for (int layer = 0; layer < 8; ++layer) {
                    async_work_group_copy(&p_linebuf0[layer*512], &p[line*512 + layer*512*512], 512, 0);
                    async_work_group_copy(&tIn_linebuf2[layer*512], &tIn[(line+1)*512 + layer*512*512], 512, 0);
                }
            }
        }
        else {
            if (line % 3 == 0) {
                for (int layer = 0; layer < 8; ++layer) {
                    async_work_group_copy(&p_linebuf0[layer*512], &p[line*512 + layer*512*512], 512, 0);
                    async_work_group_copy(&tIn_linebuf0[layer*512], &tIn[line*512 + layer*512*512], 512, 0);
                }
            }
            else if (line %3 == 1) {
                for (int layer = 0; layer < 8; ++layer) {
                    async_work_group_copy(&p_linebuf0[layer*512], &p[line*512 + layer*512*512], 512, 0);
                    async_work_group_copy(&tIn_linebuf1[layer*512], &tIn[line*512 + layer*512*512], 512, 0);
                }
            }
            else {
                for (int layer = 0; layer < 8; ++layer) {
                    async_work_group_copy(&p_linebuf0[layer*512], &p[line*512 + layer*512*512], 512, 0);
                    async_work_group_copy(&tIn_linebuf2[layer*512], &tIn[line*512 + layer*512*512], 512, 0);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

      __attribute__((xcl_pipeline_loop))
        for (int i = 0; i < 512; ++i) {
            int c = i + line * 512;

            int W = (i == 0)         ? c : c - 1;
            int E = (i == 512-1)     ? c : c + 1;
            int N = (line == 0)      ? c : c - 512;
            int S = (line == 512-1)  ? c : c + 512;

            float tIn_W, tIn_E, tIn_N, tIn_S;
            float temp1, temp2, temp3;

            if (line % 3 == 0) {
                tIn_W = tIn_linebuf2[W];
                tIn_E = tIn_linebuf2[E];
                tIn_N = tIn_linebuf1[N];
                tIn_S = tIn_linebuf0[S];

                temp1 = temp2 = tIn_linebuf2[c];
                temp3 = tIn_linebuf2[c+512];
                tOut_linebuf0[c] = cc * temp2 + cw * tIn_W + ce * tIn_E + cs * tIn_S + cn * tIn_N + cb * temp1 + ct * temp3 + sdc * p_linebuf0[c] + ct * amb_temp;
                
                c += 512;
                W += 512;
                E += 512;
                N += 512;
                S += 512;
                tIn_W = tIn_linebuf2[W];
                tIn_E = tIn_linebuf2[E];
                tIn_N = tIn_linebuf1[N];
                tIn_S = tIn_linebuf0[S];
          
                __attribute__ ((opencl_unroll_hint(6)))
                for (int k = 1; k < 7-1; ++k) {
                    temp1 = temp2;
                    temp2 = temp3;
                    temp3 = tIn_linebuf2[c+512];
                    tOut_linebuf0[c] = cc * temp2 + cw * tIn_W + ce * tIn_E + cs * tIn_S + cn * tIn_N + cb * temp1 + ct * temp3 + sdc * p_linebuf0[c] + ct * amb_temp;

                    c += 512;
                    W += 512;
                    E += 512;
                    N += 512;
                    S += 512;
                    tIn_W = tIn_linebuf2[W];
                    tIn_E = tIn_linebuf2[E];
                    tIn_N = tIn_linebuf1[N];
                    tIn_S = tIn_linebuf0[S];
                }

                temp1 = temp2;
                temp2 = temp3;
                tOut_linebuf0[c] = cc * temp2 + cw * tIn_W + ce * tIn_E + cs * tIn_S + cn * tIn_N + cb * temp1 + ct * temp3 + sdc * p_linebuf0[c] + ct * amb_temp;
            }
            else if (line % 3 == 1) {
                tIn_W = tIn_linebuf0[W];
                tIn_E = tIn_linebuf0[E];
                tIn_N = tIn_linebuf1[N];
                tIn_S = tIn_linebuf2[S];

                temp1 = temp2 = tIn_linebuf0[c];
                temp3 = tIn_linebuf0[c+512];
                tOut_linebuf0[c] = cc * temp2 + cw * tIn_W + ce * tIn_E + cs * tIn_S + cn * tIn_N + cb * temp1 + ct * temp3 + sdc * p_linebuf0[c] + ct * amb_temp;
                
                c += 512;
                W += 512;
                E += 512;
                N += 512;
                S += 512;
                tIn_W = tIn_linebuf0[W];
                tIn_E = tIn_linebuf0[E];
                tIn_N = tIn_linebuf1[N];
                tIn_S = tIn_linebuf2[S];          

                __attribute__ ((opencl_unroll_hint(6)))
                for (int k = 1; k < 7-1; ++k) {
                    temp1 = temp2;
                    temp2 = temp3;
                    temp3 = tIn_linebuf0[c+512];
                    tOut_linebuf0[c] = cc * temp2 + cw * tIn_W + ce * tIn_E + cs * tIn_S + cn * tIn_N + cb * temp1 + ct * temp3 + sdc * p_linebuf0[c] + ct * amb_temp;

                    c += 512;
                    W += 512;
                    E += 512;
                    N += 512;
                    S += 512;
                    tIn_W = tIn_linebuf0[W];
                    tIn_E = tIn_linebuf0[E];
                    tIn_N = tIn_linebuf1[N];
                    tIn_S = tIn_linebuf2[S];
                }

                temp1 = temp2;
                temp2 = temp3;
                tOut_linebuf0[c] = cc * temp2 + cw * tIn_W + ce * tIn_E + cs * tIn_S + cn * tIn_N + cb * temp1 + ct * temp3 + sdc * p_linebuf0[c] + ct * amb_temp;
            }
            else if (line %3 == 2) {
                tIn_W = tIn_linebuf1[W];
                tIn_E = tIn_linebuf1[E];
                tIn_N = tIn_linebuf0[N];
                tIn_S = tIn_linebuf2[S];

                temp1 = temp2 = tIn_linebuf1[c];
                temp3 = tIn_linebuf1[c+512];
                tOut_linebuf0[c] = cc * temp2 + cw * tIn_W + ce * tIn_E + cs * tIn_S + cn * tIn_N + cb * temp1 + ct * temp3 + sdc * p_linebuf0[c] + ct * amb_temp;
                
                c += 512;
                W += 512;
                E += 512;
                N += 512;
                S += 512;
                tIn_W = tIn_linebuf1[W];
                tIn_E = tIn_linebuf1[E];
                tIn_N = tIn_linebuf0[N];
                tIn_S = tIn_linebuf2[S];

                __attribute__ ((opencl_unroll_hint(6)))
                for (int k = 1; k < 7-1; ++k) {
                    temp1 = temp2;
                    temp2 = temp3;
                    temp3 = tIn_linebuf1[c+512];
                    tOut_linebuf0[c] = cc * temp2 + cw * tIn_W + ce * tIn_E + cs * tIn_S + cn * tIn_N + cb * temp1 + ct * temp3 + sdc * p_linebuf0[c] + ct * amb_temp;

                    c += 512;
                    W += 512;
                    E += 512;
                    N += 512;
                    S += 512;
                    tIn_W = tIn_linebuf1[W];
                    tIn_E = tIn_linebuf1[E];
                    tIn_N = tIn_linebuf0[N];
                    tIn_S = tIn_linebuf2[S];
                }

                temp1 = temp2;
                temp2 = temp3;
                tOut_linebuf0[c] = cc * temp2 + cw * tIn_W + ce * tIn_E + cs * tIn_S + cn * tIn_N + cb * temp1 + ct * temp3 + sdc * p_linebuf0[c] + ct * amb_temp;
            }
  
        }
        for (int layer = 0; layer < 8; ++layer) {
            async_work_group_copy(&tOut[line*512 + layer*512*512], &tOut_linebuf0[layer*512], 512, 0);
        }
    }

    return;
}
