#ifndef GPUCONST_CUH_
#define GPUCONST_CUH_

__constant__ float c_coef[5]; /* coefficients for 8th order fd */
__constant__ int c_nx;        /* x dim */
__constant__ int c_ny;        /* y dim */
__constant__ int c_nr;        /* num of receivers */
__constant__ int c_nxy;       /* total number of elements in the snap array (border included)*/
__constant__ int c_nb;        /* border size */
__constant__ int c_nt;        /* time steps */
__constant__ int c_dt;        /* time steps */
__constant__ float c_dt2;  /* dt2 / dx2 for fd*/
__constant__ float c_one_dx2;  /* dt2 / dx2 for fd*/
__constant__ float c_one_dy2;  /* dt2 / dx2 for fd*/

#endif
