__kernel void hotspotOpt1(__global float *p, 
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

  __local float p_local[8];
  __local float tIn_local[5][8];
  __local float tOut_local[8];
  __local int pst_local[5];
 
  int i = get_global_id(0);
  int j = get_global_id(1);
  int c = pst_local[0] = i + j * nx;
  int xy = nx * ny;

  int W = pst_local[1] = (i == 0)        ? c : c - 1;
  int E = pst_local[2] = (i == nx-1)     ? c : c + 1;
  int N = pst_local[3] = (j == 0)        ? c : c - nx;
  int S = pst_local[4] = (j == ny-1)     ? c : c + nx;

  // read the required data to local memory
  int index = c;
  for(int m = 0; m < 8; m++) {
      p_local[m] = p[index];
      index += xy;
  }

  index = c;
  for(int m = 0; m < 8; m++) {
      tOut_local[m] = tOut[index];
      index += xy;
  }

  for(int n = 0; n < 5; n++) {
      index = pst_local[n];
      for(int m = 0; m < 8; m++) {
          tIn_local[n][m] = tIn[index];
          index += xy;
      }
  }

  float temp1, temp2, temp3;
  // calculate layer 0-7
  for (int k = 0; k < nz; ++k) {
      temp1 = (k == 0) ? tIn_local[0][k] : tIn_local[0][k-1];
      temp2 = tIn_local[0][k];
      temp3 = (k == (nz-1)) ? tIn_local[0][k] : tIn_local[0][k+1];
      tOut_local[k] = cc * temp2 + cw * tIn_local[1][k] + ce * tIn_local[2][k] + cs * tIn_local[4][k]
        + cn * tIn_local[3][k] + cb * temp1 + ct * temp3 + sdc * p_local[k] + ct * amb_temp;
  }

  // write the data back to local memory
  index = c;
  for(int m = 0; m < 8; m++) {
      tOut[index] = tOut_local[m];
      index += xy;
  }
  return;
}


