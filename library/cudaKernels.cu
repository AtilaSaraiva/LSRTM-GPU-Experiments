#include "btree.cuh"
#include <stdio.h>
#include "gpuconst.cuh"
#include "cuda.h"
#include "cuda_runtime.h"

__global__ void taper_gpu (float *d_tapermask, float *campo)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int gid = gx * c_ny + gy;

    if(gid < c_nxy){
        //campo[gid] = float(1);
        campo[gid] *= d_tapermask[gid];
    }
}

__global__ void frequency (int nFreq, float df, float *DFTconstTerm)
{
    unsigned int iw = blockIdx.x * blockDim.x + threadIdx.x;

    if(iw < nFreq){
        DFTconstTerm[iw] = 2 * PI * iw / float(c_nt);
    }
}

__global__ void nucleoDFT(int nFreq, float *DFTconstTerm, float *RealKernel, float *ImagKernel)
{
    unsigned int iw = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int it = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int gid = iw * c_nt + it;

    if (iw < nFreq && it < c_nt){
        RealKernel[gid] = cosf(DFTconstTerm[iw] * it);
        ImagKernel[gid] = sinf(DFTconstTerm[iw] * it);
    }
}

__host__ void hostfreqKernelImg(int nb, int nt, int nzb, int nxb, int it, int iw, float *RealKernel, float *ImagKernel,
        float *sourceRealPart, float *sourceImagPart, float *receptRealPart, float *receptImagPart,
        float *h_ps1, float *h_pr1, float *h_rtm)
{
    int nz = nzb - 2 * nb;
    int nx = nxb - 2 * nb;
    int idx;
    for (int j=0; j < nxb; j++)
    {
        for (int i=0; i < nzb; i++)
        {
            if (i >= nb && i < nzb - nb && j >= nb && j < nxb - nb)
            {
                //std::cerr<<"i,j ="<<i<<","<<j<<std::endl;
                idx = iw * nz * nx + (j - nb) * nz + i - nb;
                sourceRealPart[idx] += h_ps1[j * nzb + i] * RealKernel[iw * nt + it];
                sourceImagPart[idx] += h_ps1[j * nzb + i] * ImagKernel[iw * nt + it];
                receptRealPart[idx] += h_pr1[j * nzb + i] * RealKernel[iw * nt + it];
                receptImagPart[idx] += h_pr1[j * nzb + i] * ImagKernel[iw * nt + it];
                h_rtm[j * nzb + i] += (sourceRealPart[idx] * receptRealPart[idx] - sourceImagPart[idx] * receptImagPart[idx]);
            }
            else
            {
                h_rtm[j * nzb + i] = float(0);
            }
        }
    }
}

__global__ void freqKernelImg(int it, int iw, int nt, float *DFTconstTerm, float *RealKernel, float *ImagKernel,
        float *sourceRealPart, float *sourceImagPart, float *receptRealPart, float *receptImagPart,
        float *d_ps1, float *d_pr1, float *d_rtm)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = gx * c_ny + gy;

    int nz = c_ny - 2 * c_nb;
    int nx = c_nx - 2 * c_nb;
    int fqIdx = iw * nz * nx + (gx - c_nb) * nz + gy - c_nb;

    if (gy >= c_nb && gy < c_ny - c_nb && gx >= c_nb && gx < c_nx - c_nb)
    {
        sourceRealPart[fqIdx] += d_ps1[idx] * RealKernel[iw * c_nt + it];
        sourceImagPart[fqIdx] += d_ps1[idx] * ImagKernel[iw * c_nt + it];
        receptRealPart[fqIdx] += d_pr1[idx] * RealKernel[iw * c_nt + it];
        receptImagPart[fqIdx] += d_pr1[idx] * ImagKernel[iw * c_nt + it];
        //d_rtm[idx] += (DFTconstTerm[iw] * DFTconstTerm[iw]) *(sourceRealPart * receptRealPart - sourceImagPart * receptImagPart);
        d_rtm[idx] += (sourceRealPart[fqIdx] * receptRealPart[fqIdx] - sourceImagPart[fqIdx] * receptImagPart[fqIdx]);
    }
}

__global__ void receptors(int it, int nr, int gxbeg, float *d_u1, float *d_data)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if(gx < nr){
        d_data[gx * c_nt + it] = d_u1[(gx + gxbeg + c_nb) * c_ny + c_nb + 1];
    }
}

// Add source wavelet
__global__ void kernel_add_wavelet(float *d_u, float *d_wavelet, int it, int jsrc, int isrc)
{
    /*
    d_u             :pointer to an array on device where to add source term
    d_wavelet       :pointer to an array on device with source signature
    it              :time step id
    */
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = gx * c_ny + gy;

    if ((gx == jsrc + c_nb) && (gy == isrc + c_nb))
    {
        d_u[idx] += d_wavelet[it];
    }
}

// Add a whole shot gather
__global__ void kernel_add_seismicdata(int it, int nr, int gxbeg, float *d_u1, float *d_seisdata)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if(gx < nr){
        d_u1[(gx + gxbeg + c_nb) * c_ny + c_nb] += d_seisdata[gx * c_nt + it];
    }
}

__global__ void kernel_image_condition(float *d_u, float *d_q, float *d_rtm)
{
    /*
    d_u             :pointer to an array on device where to add source term
    d_wavelet       :pointer to an array on device with source signature
    it              :time step id
    */
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = gx * c_ny + gy;

    if (gy < c_ny && gx < c_nx)
    {
        d_rtm[idx] += d_u[idx] * d_q[idx];
    }
}


__global__ void kernel_add_sourceArray(float *d_u, float *d_sourceArray)
{
    /*
    d_u             :pointer to an array on device where to add source term
    d_wavelet       :pointer to an array on device with source signature
    it              :time step id
    */
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = gx * c_ny + gy;

    unsigned int gxWoutBord = gx - c_nb;
    unsigned int gyWoutBord = gy - c_nb;
    unsigned int nyWoutBord = c_ny - 2 * c_nb;

    if (gy >= c_nb && gy < c_ny - c_nb && gx >= c_nb && gx < c_nx - c_nb)
    {
        d_u[idx] += d_sourceArray[gxWoutBord * nyWoutBord + gyWoutBord];
    }
}

__global__ void kernel_applySourceArray(float dt, float *d_reflectivity, float *d_pField, float *d_vel, float *d_q)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = gx * c_ny + gy;

    unsigned int gxWoutBord = gx - c_nb;
    unsigned int gyWoutBord = gy - c_nb;
    unsigned int nyWoutBord = c_ny - 2 * c_nb;

    float v_dt2;
    float updValue;

    if (gy >= c_nb && gy < c_ny - c_nb && gx >= c_nb && gx < c_nx - c_nb)
    {
        //v_dt2 = d_vel[idx] * d_vel[idx] * dt * dt;
        //updValue = -1 * v_dt2 * d_pField[idx] * d_reflectivity[gxWoutBord * nyWoutBord + gyWoutBord];
        updValue = -1 * d_pField[idx] * d_reflectivity[gxWoutBord * nyWoutBord + gyWoutBord];
        d_q[idx] += updValue;
    }
}

__global__ void kernel_applySourceArray_ver2(float dt, float *d_reflectivity, float *d_lap, float *d_vel, float *d_q)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = gx * c_ny + gy;

    unsigned int gxWoutBord = gx - c_nb;
    unsigned int gyWoutBord = gy - c_nb;
    unsigned int nyWoutBord = c_ny - 2 * c_nb;

    //float v_dt2;
    float updValue;

    if (gy >= c_nb && gy < c_ny - c_nb && gx >= c_nb && gx < c_nx - c_nb)
    {
        //v_dt2 = d_vel[idx] * d_vel[idx] * dt * dt;
        //updValue = -1 * v_dt2 * d_pField[idx] * d_reflectivity[gxWoutBord * nyWoutBord + gyWoutBord];
        updValue = d_lap[idx] * d_reflectivity[gxWoutBord * nyWoutBord + gyWoutBord];
        d_q[idx] += updValue;
    }
}


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


// FD kernel
__global__ void kernel_2dfd(float *d_u1, float *d_u2, float *d_vp)
{
    // save model dims in registers as they are much faster
    const int nx = c_nx;
    const int ny = c_ny;

    // FD coefficient dt2 / dx2
    const float dt2 = c_dt2;
    const float one_dx2 = c_one_dx2;
    const float one_dy2 = c_one_dy2;

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
    __shared__ float s_u1[SDIMY][SDIMX];
    __shared__ float s_u2[SDIMY][SDIMX];
    __shared__ float s_vp[SDIMY][SDIMX];

    // If thread points into the physical domain
    if ((gx < nx) && (gy < ny))
    {
        // Copy regions from gmem into smem
        //       gmem, smem,  block, shared, global, dims
        set_halo(d_u1, s_u1, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_u2, s_u2, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_vp, s_vp, tx, ty, sx, sy, gx, gy, nx, ny);
        __syncthreads();

        // Central point of fd stencil, o o o o x o o o o
        float du2_xx = c_coef[0] * s_u2[sy][sx];
        float du2_yy = c_coef[0] * s_u2[sy][sx];

#pragma unroll
        for (int d = 1; d <= 4; d++)
        {
            du2_xx += c_coef[d] * (s_u2[sy][sx - d] + s_u2[sy][sx + d]);
            du2_yy += c_coef[d] * (s_u2[sy - d][sx] + s_u2[sy + d][sx]);
        }
        // Second order wave equation
        d_u1[idx] = 2.0 * s_u2[sy][sx] - s_u1[sy][sx] + s_vp[sy][sx] * s_vp[sy][sx] * (du2_xx * one_dx2 + du2_yy * one_dy2) * dt2;
        //d_u1[idx] = du2_xx;

        __syncthreads();
    }
}

__global__ void kernel_2dfd_ver2(float *d_lap, float *d_u1, float *d_u2, float *d_vp)
{
    // save model dims in registers as they are much faster
    const int nx = c_nx;
    const int ny = c_ny;

    // FD coefficient dt2 / dx2
    const float dt2 = c_dt2;
    const float one_dx2 = c_one_dx2;
    const float one_dy2 = c_one_dy2;

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
    __shared__ float s_u1[SDIMY][SDIMX];
    __shared__ float s_u2[SDIMY][SDIMX];
    __shared__ float s_vp[SDIMY][SDIMX];

    // If thread points into the physical domain
    if ((gx < nx) && (gy < ny))
    {
        // Copy regions from gmem into smem
        //       gmem, smem,  block, shared, global, dims
        set_halo(d_u1, s_u1, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_u2, s_u2, tx, ty, sx, sy, gx, gy, nx, ny);
        set_halo(d_vp, s_vp, tx, ty, sx, sy, gx, gy, nx, ny);
        __syncthreads();

        // Central point of fd stencil, o o o o x o o o o
        float du2_xx = c_coef[0] * s_u2[sy][sx];
        float du2_yy = c_coef[0] * s_u2[sy][sx];

#pragma unroll
        for (int d = 1; d <= 4; d++)
        {
            du2_xx += c_coef[d] * (s_u2[sy][sx - d] + s_u2[sy][sx + d]);
            du2_yy += c_coef[d] * (s_u2[sy - d][sx] + s_u2[sy + d][sx]);
        }
        // Second order wave equation
        d_lap[idx] = du2_xx * one_dx2 + du2_yy * one_dy2;
        d_u1[idx] = 2.0 * s_u2[sy][sx] - s_u1[sy][sx] + s_vp[sy][sx] * s_vp[sy][sx] * (du2_xx * one_dx2 + du2_yy * one_dy2) * dt2;
        //d_u1[idx] = du2_xx;

        __syncthreads();
    }
}
