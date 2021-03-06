#include <iostream>
#include <rsf.hh>
#include "cuda.h"
#include "cuda_runtime.h"
#include "btree.cuh"
#include "gpuconst.cuh"

#include "cudaKernels.cu"

using namespace std;

void test_kernel_add_sourceArray(float *d_reflectivity, geometry param, dim3 grid, dim3 block)
{
    float *d_zeros;
    CHECK(cudaMalloc((void **)&d_zeros, param.nbytes))       /* wavefield at t-2 */
    CHECK(cudaMemset(d_zeros, 0, param.nbytes))
    kernel_add_sourceArray<<<grid,block>>>(d_zeros, d_reflectivity);

    float *h_zeros = new float[param.nbxy];
    CHECK(cudaMemcpy(h_zeros, d_zeros, param.nbytes, cudaMemcpyDeviceToHost));

    FILE *f_test = fopen("test_kernel_add_sourceArray", "w");

    fwrite(h_zeros, sizeof(float), param.nbxy, f_test);
    fclose(f_test);
}

void rtm(geometry param, velocity h_model, float *h_wavelet, float *h_tapermask, seismicData h_seisData, sf_file Fdata)
{

    float dt2 = (h_seisData.timeStep * h_seisData.timeStep);
    float one_dx2 = float(1) / (param.modelDx * param.modelDx);
    float one_dy2 = float(1) / (param.modelDy * param.modelDy);
    size_t dbytes = param.nReceptors * h_seisData.timeSamplesNt * sizeof(float);
    size_t tbytes = h_seisData.timeSamplesNt * sizeof(float);

    // Allocate memory on device
    printf("Allocate and copy memory on the device...\n");
    float *d_u1, *d_u2, *d_vp, *d_wavelet, *d_tapermask, *d_rtm, *d_q, *d_seisdata;
    CHECK(cudaMalloc((void **)&d_u1, param.nbytes));       /* wavefield at t-2 */
    CHECK(cudaMalloc((void **)&d_q, param.nbytes));     /* wavefield at t-2 */
    CHECK(cudaMalloc((void **)&d_u2, param.nbytes));       /* wavefield at t-1 */
    CHECK(cudaMalloc((void **)&d_vp, param.nbytes));      /* velocity model */
    CHECK(cudaMalloc((void **)&d_rtm, param.nbytes));       /* rtm image */
    CHECK(cudaMalloc((void **)&d_wavelet, tbytes)); /* source term for each time step */
    CHECK(cudaMalloc((void **)&d_tapermask, param.nbytes));
    CHECK(cudaMalloc((void **)&d_seisdata, dbytes));

    // Fill allocated memory with a value
    CHECK(cudaMemset(d_rtm, 0, param.nbytes))

    // Copy arrays from host to device
    CHECK(cudaMemcpy(d_vp, h_model.extVelField, param.nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tapermask, h_tapermask, param.nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wavelet, h_wavelet, tbytes, cudaMemcpyHostToDevice));

    // Copy constants to device constant memory
    float coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpyToSymbol(c_coef, coef, 5 * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_nx, &param.modelNxBorder, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_ny, &param.modelNyBorder, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nr, &param.nReceptors, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nxy, &param.nbxy, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nb, &param.taperBorder, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nt, &h_seisData.timeSamplesNt, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_dt2, &dt2, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_one_dx2, &one_dx2, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_one_dy2, &one_dy2, sizeof(float)));
    //printf("\t%f MB\n", (4 * param.nbytes + tbytes)/1024/1024);
    //printf("OK\n");

    // Print out specs of the main GPU
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    //printf("GPU0:\t%s\t%d.%d:\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    //printf("\t%lu GB:\t total Global memory (gmem)\n", deviceProp.totalGlobalMem / 1024 / 1024 / 1000);
    //printf("\t%lu MB:\t total Constant memory (cmem)\n", deviceProp.totalConstMem / 1024);
    //printf("\t%lu MB:\t total Shared memory per block (smem)\n", deviceProp.sharedMemPerBlock / 1024);
    //printf("\t%d:\t total threads per block\n", deviceProp.maxThreadsPerBlock);
    //printf("\t%d:\t total registers per block\n", deviceProp.regsPerBlock);
    //printf("\t%d:\t warp size\n", deviceProp.warpSize);
    //printf("\t%d x %d x %d:\t max dims of block\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    //printf("\t%d x %d x %d:\t max dims of grid\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    CHECK(cudaSetDevice(0));

    // Print out CUDA domain partitioning info
    //printf("CUDA:\n");
    //printf("\t%i x %i\t:block dim\n", BDIMY, BDIMX);
    //printf("\t%i x %i\t:shared dim\n", SDIMY, SDIMX);
    //printf("CFL:\n");
    //printf("\t%f\n", _vp * h_seisData.timeStep / dx);

    // Setup CUDA run
    dim3 block(BDIMX, BDIMY);
    dim3 grid((param.modelNxBorder + block.x - 1) / block.x, (param.modelNyBorder + block.y - 1) / block.y);

    int snapStep = 1;
    size_t snapTotalSize = ((h_seisData.timeSamplesNt - 1)  / snapStep + 1) * param.nbxy;
    float* snapsBuffer = new float[snapTotalSize];

    // MAIN LOOP
    for(int shot=0; shot<param.nShots; shot++){
        cerr<<"\nShot "<<shot<<" param.firstReceptorPos = "<<param.firstReceptorPos<<", param.srcPosX = "<<param.srcPosX<<", param.srcPosY = "<<param.srcPosY<<
            ", param.incShots = "<<param.incShots<<"\n"<<endl;

        CHECK(cudaMemcpy(d_seisdata, &h_seisData.seismogram[shot * param.nReceptors * h_seisData.timeSamplesNt], dbytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_u1, 0, param.nbytes))
        CHECK(cudaMemset(d_u2, 0, param.nbytes))

        float *d_u3;
        printf("Time loop...\n");
        for (int it = 0; it < h_seisData.timeSamplesNt; it++)
        {
            taper_gpu<<<grid,block>>>(d_tapermask, d_u1);
            taper_gpu<<<grid,block>>>(d_tapermask, d_u2);

            // These kernels are in the same stream so they will be executed one by one
            kernel_add_wavelet<<<grid, block>>>(d_u2, d_wavelet, it, param.srcPosX, param.srcPosY);
            kernel_2dfd<<<grid, block>>>(d_u1, d_u2, d_vp);

            // Exchange time steps
            d_u3 = d_u1;
            d_u1 = d_u2;
            d_u2 = d_u3;

            //if(shot == 0 && it % snapStep == 0){
            CHECK(cudaMemcpy(&snapsBuffer[it/snapStep * param.nbxy], d_u3, param.nbytes, cudaMemcpyDeviceToHost));
            //}

            // Save snapshot every h_wavelet.snapStep iterations
            if ((it % 50 == 0))
            {
                //printf("%i/%i\n", it+1, h_seisData.timeSamplesNt);
                cerr<<it+1<<"/"<<h_seisData.timeSamplesNt<<endl;
                //saveSnapshotIstep(it, d_u3, param.modelNxBorder, param.modelNyBorder, "u3", shot);
            }
        }

        CHECK(cudaMemset(d_u1, 0, param.nbytes))
        CHECK(cudaMemset(d_u2, 0, param.nbytes))

        for (int it = h_seisData.timeSamplesNt - 1; it >= 0; it--)
        {
            taper_gpu<<<grid,block>>>(d_tapermask, d_u1);
            taper_gpu<<<grid,block>>>(d_tapermask, d_u2);

            // These kernels are in the same stream so they will be executed one by one
            kernel_2dfd<<<grid, block>>>(d_u1, d_u2, d_vp);
            kernel_add_seismicdata<<<(param.nReceptors + 32) / 32, 32>>>(it, param.nReceptors, param.firstReceptorPos, d_u1, d_seisdata);

            CHECK(cudaMemcpy(d_q, &snapsBuffer[it * param.nbxy], param.nbytes, cudaMemcpyHostToDevice));
            kernel_image_condition<<<grid,block>>>(d_u1, d_q, d_rtm);

            // Exchange time steps
            d_u3 = d_u1;
            d_u1 = d_u2;
            d_u2 = d_u3;

            if ((it % 50 == 0))
            {
                float* h_aux = new float[param.nbxy];
                cerr<<it+1<<"/"<<h_seisData.timeSamplesNt<<endl;
                //char fname[32];
                //sprintf(fname, "snap/upgoing_%s_s%i_%i_%i_%i", "u3", shot, it, param.modelNyBorder, param.modelNxBorder);
                //FILE *snap = fopen(fname, "w");
                //CHECK(cudaMemcpy(h_aux, d_q, param.nbytes, cudaMemcpyDeviceToHost));
                //fwrite(h_aux, sizeof(float), param.nbxy, snap);
                //fflush(stdout);
                //fclose(snap);
                //sprintf(fname, "snap/downgoing_%s_s%i_%i_%i_%i", "u3", shot, it, param.modelNyBorder, param.modelNxBorder);
                //FILE *snap1 = fopen(fname, "w");
                //CHECK(cudaMemcpy(h_aux, d_u3, param.nbytes, cudaMemcpyDeviceToHost));
                //fwrite(h_aux, sizeof(float), param.nbxy, snap);
                //fflush(stdout);
                //fclose(snap1);
                //sprintf(fname, "snap/product_%s_s%i_%i_%i_%i", "u3", shot, it, param.modelNyBorder, param.modelNxBorder);
                //FILE *snap2 = fopen(fname, "w");
                //CHECK(cudaMemcpy(h_aux, d_rtm, param.nbytes, cudaMemcpyDeviceToHost));
                //fwrite(h_aux, sizeof(float), param.nbxy, snap);
                //fflush(stdout);
                //fclose(snap2);
            }
        }

        param.firstReceptorPos += param.incRec;
        param.srcPosX += param.incShots;
    }

    float* h_rtm = new float[param.nbxy];
    //kernel_lap_filter<<<grid, block>>>(d_rtm);
    CHECK(cudaMemcpy(h_rtm, d_rtm, param.nbytes, cudaMemcpyDeviceToHost));
    for(int j=param.taperBorder; j<param.modelNx + param.taperBorder; j++){
        sf_floatwrite(&h_rtm[j*param.modelNyBorder + param.taperBorder], param.modelNy, Fdata);
    }
    printf("OK\n");
    delete h_rtm;


    CHECK(cudaGetLastError());


    delete snapsBuffer;
    CHECK(cudaFree(d_u1));
    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_u2));
    CHECK(cudaFree(d_tapermask));
    CHECK(cudaFree(d_seisdata));
    CHECK(cudaFree(d_vp));
    CHECK(cudaFree(d_wavelet));
    //printf("OK saigo\n");
    CHECK(cudaDeviceReset());
}
