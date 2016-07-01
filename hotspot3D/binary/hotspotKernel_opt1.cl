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

  // calculate layer 0
  float temp1, temp2, temp3;
  temp1 = temp2 = tIn_local[0][0]; // center & layer 0
  temp3 = tIn_local[0][1];  // center & layer 1
  tOut_local[0] = cc * temp2 + cw * tIn_local[1][0] + ce * tIn_local[2][0] + cs * tIn_local[4][0]
    + cn * tIn_local[3][0] + cb * temp1 + ct * temp3 + sdc * p_local[0] + ct * amb_temp;

  // calculate layer 0-6
  for (int k = 1; k < nz-1; ++k) {
      temp1 = temp2;
      temp2 = temp3;
      temp3 = tIn_local[0][k+1];
      tOut_local[k] = cc * temp2 + cw * tIn_local[1][k] + ce * tIn_local[2][k] + cs * tIn_local[4][k]
        + cn * tIn_local[3][k] + cb * temp1 + ct * temp3 + sdc * p_local[k] + ct * amb_temp;
  }

  // calculate layer 7
  temp1 = temp2;
  temp2 = temp3;
  tOut_local[7] = cc * temp2 + cw * tIn_local[1][7] + ce * tIn_local[2][7] + cs * tIn_local[4][7]
    + cn * tIn_local[3][7] + cb * temp1 + ct * temp3 + sdc * p_local[7] + ct * amb_temp;

  // write the data back to local memory
  index = c;
  for(int m = 0; m < 8; m++) {
      p[index] = p_local[m];
      index += xy;
  }

  index = c;
  for(int m = 0; m < 8; m++) {
      tOut[index] = tOut_local[m];
      index += xy;
  }

  for(int n = 0; n < 5; n++) {
      index = pst_local[n];
      for(int m = 0; m < 8; m++) {
          tIn[index] = tIn_local[n][m];
          index += xy;
      }
  }
  return;
}


