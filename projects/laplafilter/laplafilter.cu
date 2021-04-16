#include <rsf.hh>
#include <iostream>
#include <string>
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"

#include "cuda.h"
#include "cuda_runtime.h"

__constant__ int c_nx;        /* x dim */
__constant__ int c_nz;        /* z dim */
__constant__ float c_one_dx2;  /* dt2 / dx2 for fd*/
__constant__ float c_one_dz2;  /* dt2 / dx2 for fd*/
__constant__ float c_coef[5]; /* coefficients for 8th order fd */


#define PI 3.14159265359

// Padding for FD scheme
#define HALO 4
#define HALO2 8

// FD stencil coefficients
#define a0  -2.8472222f
#define a1   1.6000000f
#define a2  -0.2000000f
#define a3   0.0253968f
#define a4  -0.0017857f

// Block dimensions
#define BDIMX 32
#define BDIMY 32

// Shared memory tile dimenstions
#define SDIMX BDIMX + HALO2
#define SDIMY BDIMY + HALO2

#define CHECK(call)                                                \
{                                                              \
    cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                  \
    {                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error,       \
                cudaGetErrorString(error));                    \
    }                                                          \
}

using namespace std;

__device__ void set_halo(float *global, float shared[][SDIMX], int tx, int ty, int sx, int sy, int gx, int gy, int nx, int ny)
{
    /*
    global      :pointer to an array in global memory (gmem)
    shared      :2D array in shared device memory
    tx, ty      :thread id's in a block
    sx, sy      :thread id's in a shared memory tile
    gx, gy      :thread id's in the entire computational domain
    */

    // Each thread copies one value from gmem into smem
    shared[sy][sx] = global[gx * ny + gy];

    // Populate halo regions in smem for left, right, top and bottom boundaries of a block
    // if thread near LEFT border of a block
    if (tx < HALO)
    {
        // if global left
        if (gx < HALO)
        {
            // reflective boundary
            shared[sy][sx - HALO] = 0.0;
        }
        else
        {
            // if block left
            shared[sy][sx - HALO] = global[(gx - HALO) * ny + gy];
        }
    }
    // if thread near RIGHT border of a block
    if ((tx >= (BDIMX - HALO)) || ((gx + HALO) >= nx))
    {
        // if global right
        if ((gx + HALO) >= nx)
        {
            // reflective boundary
            shared[sy][sx + HALO] = 0.0;
        }
        else
        {
            // if block right
            shared[sy][sx + HALO] = global[(gx + HALO) * ny + gy];
        }
    }

    // if thread near BOTTOM border of a block
    if (ty < HALO)
    {
        // if global bottom
        if (gy < HALO)
        {
            // reflective boundary
            shared[sy - HALO][sx] = 0.0;
        }
        else
        {
            // if block bottom
            shared[sy - HALO][sx] = global[gx * ny + gy - HALO];
        }
    }

    // if thread near TOP border of a block
    if ((ty >= (BDIMY - HALO)) || ((gy + HALO) >= ny))
    {
        // if global top
        if ((gy + HALO) >= ny)
        {
            // reflective boundary
            shared[sy + HALO][sx] = 0.0;
        }
        else
        {
            // if block top
            shared[sy + HALO][sx] = global[gx * ny + gy + HALO];
        }
    }
}

// Lapla filter kernel
__global__ void kernel_lap_filter(float *d_rtm)
{
    // save model dims in registers as they are much faster
    const int nx = c_nx;
    const int ny = c_nz;

    // FD coefficient dt2 / dx2
    const float one_dx2 = c_one_dx2;
    const float one_dy2 = c_one_dz2;

    // Thread address (ty, tx) in a block
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    // Thread address (sy, sx) in shared memory
    const unsigned int sx = threadIdx.x + HALO;
    const unsigned int sy = threadIdx.y + HALO;

    // Thread address (gy, gx) in global memory
    const unsigned int gx = blockIdx.x * blockDim.x + tx;
    const unsigned int gy = blockIdx.y * blockDim.y + ty;

    // Global linear index
    const unsigned int idx = gx * ny + gy;

    // Allocate shared memory for a block (smem)
    __shared__ float s_rtm[SDIMY][SDIMX];

    // If thread points into the physical domain
    if ((gx < nx) && (gy < ny))
    {
        // Copy regions from gmem into smem
        //        gmem, smem,  block,  shared, global, dims
        set_halo(d_rtm, s_rtm, tx, ty, sx, sy, gx, gy, nx, ny);
        __syncthreads();

        //// Central point of fd stencil, o o o o x o o o o
        //float drtm_xx = c_coef[0] * s_rtm[sy][sx];
        //float drtm_yy = c_coef[0] * s_rtm[sy][sx];

//#pragma unroll
        //for (int d = 1; d <= 1; d++)
        //{
            //drtm_xx += c_coef[d] * (s_rtm[sy][sx - d] + s_rtm[sy][sx + d]);
            //drtm_yy += c_coef[d] * (s_rtm[sy - d][sx] + s_rtm[sy + d][sx]);
        //}

        //float drtm_yy = s_rtm[sy+1][sx] + s_rtm[sy-1][sx] - 2 * s_rtm[sy][sx];

        // Second order wave equation
        //d_rtm[idx] = drtm_xx * one_dx2 + drtm_yy * one_dy2;
        d_rtm[idx] = s_rtm[sy+1][sx] - s_rtm[sy-1][sx];

        __syncthreads();
    }
}


int main(int argc, char *argv[])
{
    /* Main program that reads and writes data and read input variables */
    bool verb;
    sf_init(argc,argv); // init RSF
    if(! sf_getbool("verb",&verb)) verb=0;

    sf_file Fimg = sf_input("img");

    int nz, nx;
    float dz, dx;
    sf_histint(Fimg, "n1",&nz);
    sf_histint(Fimg, "n2",&nx);
    sf_histfloat(Fimg, "d1",&dz);
    sf_histfloat(Fimg, "d2",&dx);

    int nxz = nz * nx;
    size_t nxzbytes = nxz * sizeof(float);

    float* img = new float[nxz];
    sf_floatread(img, nxz, Fimg);

    sf_file Flap = sf_output("lap");
    sf_putint(Flap, "n1",nz);
    sf_putint(Flap, "n2",nx);
    sf_putfloat(Flap, "d1",dz);
    sf_putfloat(Flap, "d2",dx);
    float* lap = new float[nxz];

    float one_dx2 = float(1) / (dx * dx);
    float one_dz2 = float(1) / (dz * dz);


    float coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpyToSymbol(c_coef, coef, 5 * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_nx, &nx, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nz, &nz, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_one_dx2, &one_dx2, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_one_dz2, &one_dz2, sizeof(float)));

    float *d_img;
    CHECK(cudaMalloc((void **)&d_img, nxzbytes));       /* rtm image */
    CHECK(cudaMemcpy(d_img, img, nxzbytes, cudaMemcpyHostToDevice));

    // Setup CUDA run
    dim3 block(BDIMX, BDIMY);
    dim3 grid((nx + block.x - 1) / block.x, (nz + block.y - 1) / block.y);

    CHECK(cudaSetDevice(0));

    kernel_lap_filter<<<grid, block>>>(d_img);

    CHECK(cudaMemcpy(lap, d_img, nxzbytes, cudaMemcpyDeviceToHost));

    cerr<<"lap[160 * nz + 32] = "<<lap[160 * nz + 32]<<endl;
    sf_floatwrite(lap, nxz, Flap);

    delete[] img;
    delete[] lap;
    CHECK(cudaFree(d_img));

    sf_close();
    return 0;
}
