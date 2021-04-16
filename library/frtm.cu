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

void frtm(geometry param, velocity h_model, float *h_wavelet, float *h_tapermask, seismicData h_seisData, sf_file Fdata)
{
    float dt2 = (h_seisData.timeStep * h_seisData.timeStep);
    float one_dx2 = float(1) / (param.modelDx * param.modelDx);
    float one_dy2 = float(1) / (param.modelDy * param.modelDy);
    size_t dbytes = param.nReceptors * h_seisData.timeSamplesNt * sizeof(float);
    size_t tbytes = h_seisData.timeSamplesNt * sizeof(float);
    //int nFreq = 200;
    float fmax = 45;
    float fmin = 1;
    float df = float(1) / (h_seisData.timeSamplesNt * h_seisData.timeStep);
    int nFreq = (int)(fmax - fmin) / df;

    cerr<<"nw = "<<nFreq<<endl;
    cerr<<"df = "<<df<<endl;
    cerr<<"dt = "<<h_seisData.timeStep<<endl;

    // Allocate memory on device
    printf("Allocate and copy memory on the device...\n");
    float *d_pr1, *d_pr2, *d_ps1, *d_ps2;
    float *d_vp, *d_wavelet, *d_tapermask;
    float *d_rtm, *d_q, *d_seisdata, *d_DFTconst;
    float *d_RealKernel, *d_ImagKernel;


    CHECK(cudaMalloc((void **)&d_ps1, param.nbytes));       /* wavefield at t-2 */
    CHECK(cudaMalloc((void **)&d_ps2, param.nbytes));       /* wavefield at t-1 */
    CHECK(cudaMalloc((void **)&d_pr1, param.nbytes));       /* wavefield at t-2 */
    CHECK(cudaMalloc((void **)&d_pr2, param.nbytes));       /* wavefield at t-1 */
    CHECK(cudaMalloc((void **)&d_q, param.nbytes));       /* wavefield at t-2 */
    CHECK(cudaMalloc((void **)&d_vp, param.nbytes));      /* velocity model */
    CHECK(cudaMalloc((void **)&d_rtm, param.nbytes));       /* rtm image */
    CHECK(cudaMalloc((void **)&d_wavelet, tbytes)); /* source term for each time step */
    CHECK(cudaMalloc((void **)&d_DFTconst, tbytes)); /* source term for each time step */
    CHECK(cudaMalloc((void **)&d_RealKernel, nFreq * h_seisData.timeSamplesNt * sizeof(float))); /* source term for each time step */
    CHECK(cudaMalloc((void **)&d_ImagKernel, nFreq * h_seisData.timeSamplesNt * sizeof(float))); /* source term for each time step */
    CHECK(cudaMalloc((void **)&d_tapermask, param.nbytes));
    CHECK(cudaMalloc((void **)&d_seisdata, dbytes));


    float *sourceRealPart, *sourceImagPart;
    float *receptRealPart, *receptImagPart;
    CHECK(cudaMallocManaged(&sourceRealPart, param.nxy * nFreq * sizeof(float)));
    CHECK(cudaMallocManaged(&sourceImagPart, param.nxy * nFreq * sizeof(float)));
    CHECK(cudaMallocManaged(&receptRealPart, param.nxy * nFreq * sizeof(float)));
    CHECK(cudaMallocManaged(&receptImagPart, param.nxy * nFreq * sizeof(float)));
    CHECK(cudaMemset(sourceRealPart, 0, param.nxy * sizeof(float)));
    CHECK(cudaMemset(sourceImagPart, 0, param.nxy * sizeof(float)));
    CHECK(cudaMemset(receptRealPart, 0, param.nxy * sizeof(float)));
    CHECK(cudaMemset(receptImagPart, 0, param.nxy * sizeof(float)));


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
    CHECK(cudaMemcpyToSymbol(c_dt, &h_seisData.timeStep, sizeof(float)));
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


    dim3 grid2((nFreq + block.x - 1) / block.x, (h_seisData.timeSamplesNt + block.y - 1) / block.y);
    frequency<<<(nFreq + 32) / 32, 32>>>(nFreq, df, d_DFTconst);
    nucleoDFT<<<grid2,block>>>(nFreq, d_DFTconst, d_RealKernel, d_ImagKernel);
    //FILE *fp_realKer = fopen("realkernelDFTprotasio", "r");
    //FILE *fp_imagKer = fopen("imagkernelDFTprotasio", "r");
    //float *h_RealKernel = new float[nFreq * h_seisData.timeSamplesNt];
    //float *h_ImagKernel = new float[nFreq * h_seisData.timeSamplesNt];
    //fread(h_RealKernel, sizeof(float), nFreq * h_seisData.timeSamplesNt, fp_realKer);
    //fread(h_ImagKernel, sizeof(float), nFreq * h_seisData.timeSamplesNt, fp_imagKer);
    //CHECK(cudaMemcpy(d_RealKernel, h_RealKernel, nFreq * h_seisData.timeSamplesNt * sizeof(float), cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(d_ImagKernel, h_ImagKernel, nFreq * h_seisData.timeSamplesNt * sizeof(float), cudaMemcpyHostToDevice));


    //size_t nt2 = nFreq * h_seisData.timeSamplesNt;
    //float *aux = new float[nt2];
    //CHECK(cudaMemcpy(aux, d_RealKernel, nFreq * h_seisData.timeSamplesNt * sizeof(float), cudaMemcpyDeviceToHost));
    //float *aux2 = new float[nt2];
    //CHECK(cudaMemcpy(aux2, d_ImagKernel, nFreq * h_seisData.timeSamplesNt * sizeof(float), cudaMemcpyDeviceToHost));
    //FILE *fp_snap = fopen("realkernelDFT", "w");
    //fwrite(aux, sizeof(float), nt2, fp_snap);
    //printf("Saved the DFT kernel real part");
    //fflush(stdout);
    //fclose(fp_snap);
    //FILE *fp_snap2 = fopen("imagkernelDFT", "w");
    //fwrite(aux, sizeof(float), nt2, fp_snap2);
    //printf("Saved the DFT kernel imag part");
    //fflush(stdout);
    //fclose(fp_snap2);
    //delete aux, aux2;


    // MAIN LOOP
    for(int shot=0; shot<param.nShots; shot++){
        cerr<<"\nShot "<<shot<<" param.firstReceptorPos = "<<param.firstReceptorPos<<", param.srcPosX = "<<param.srcPosX<<", param.srcPosY = "<<param.srcPosY<<
            ", param.incShots = "<<param.incShots<<"\n"<<endl;

        CHECK(cudaMemcpy(d_seisdata, &h_seisData.seismogram[shot * param.nReceptors * h_seisData.timeSamplesNt], dbytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_ps1, 0, param.nbytes))
        CHECK(cudaMemset(d_ps2, 0, param.nbytes))
        CHECK(cudaMemset(d_pr1, 0, param.nbytes))
        CHECK(cudaMemset(d_pr2, 0, param.nbytes))

        float *d_ps3, *d_pr3;
        printf("Time loop...\n");
        for (int it = 0; it < h_seisData.timeSamplesNt; it++)
        {
            taper_gpu<<<grid,block>>>(d_tapermask, d_ps1);
            taper_gpu<<<grid,block>>>(d_tapermask, d_ps2);

            // These kernels are in the same stream so they will be executed one by one
            kernel_add_wavelet<<<grid, block>>>(d_ps2, d_wavelet, it, param.srcPosX, param.srcPosY);
            kernel_2dfd<<<grid, block>>>(d_ps1, d_ps2, d_vp);

            taper_gpu<<<grid,block>>>(d_tapermask, d_pr1);
            taper_gpu<<<grid,block>>>(d_tapermask, d_pr2);

            // These kernels are in the same stream so they will be executed one by one
            kernel_add_seismicdata<<<(param.nReceptors + 32) / 32, 32>>>(h_seisData.timeSamplesNt - it, param.nReceptors, param.firstReceptorPos, d_pr2, d_seisdata);
            kernel_2dfd<<<grid, block>>>(d_pr1, d_pr2, d_vp);

            for(int iw=0; iw < nFreq; iw++)
            {
                //auto a = "           ";
                //if(iw % 100 == 0 && it % 50 == 0) cerr<<"\rI'm making some progress "<<iw;
                freqKernelImg<<<grid,block>>>(it, iw, h_seisData.timeSamplesNt, d_DFTconst, d_RealKernel, d_ImagKernel,
                         sourceRealPart, sourceImagPart, receptRealPart, receptImagPart,
                         d_ps1, d_pr1, d_rtm);
            }

            // Exchange time steps
            d_pr3 = d_pr1;
            d_pr1 = d_pr2;
            d_pr2 = d_pr3;

            // Exchange time steps
            d_ps3 = d_ps1;
            d_ps1 = d_ps2;
            d_ps2 = d_ps3;

            // Save snapshot every h_wavelet.snapStep iterations
            if ((it % 50 == 0))
            {
                cerr<<" "<<endl;
                //printf("%i/%i\n", it+1, h_seisData.timeSamplesNt);
                cerr<<it+1<<"/"<<h_seisData.timeSamplesNt<<endl;



            }
        }

        param.firstReceptorPos += param.incRec;
        param.srcPosX += param.incShots;
    }

    float* h_rtm = new float[param.nbxy];
    CHECK(cudaMemcpy(h_rtm, d_rtm, param.nbytes, cudaMemcpyDeviceToHost));
    for(int j=param.taperBorder; j<param.modelNx + param.taperBorder; j++){
        sf_floatwrite(&h_rtm[j*param.modelNyBorder + param.taperBorder], param.modelNy, Fdata);
    }
    printf("OK\n");
    delete h_rtm;


    CHECK(cudaGetLastError());



    delete snapsBuffer;
    CHECK(cudaFree(sourceImagPart));
    CHECK(cudaFree(receptImagPart));
    CHECK(cudaFree(sourceRealPart));
    CHECK(cudaFree(receptRealPart));
    CHECK(cudaFree(d_ps1));
    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_ps2));
    CHECK(cudaFree(d_tapermask));
    CHECK(cudaFree(d_seisdata));
    CHECK(cudaFree(d_vp));
    CHECK(cudaFree(d_wavelet));
    //printf("OK saigo\n");
    CHECK(cudaDeviceReset());
}

                //// DELETE AFTER, only for testing
                //CHECK(cudaMemcpy(aux_pr, d_pr3, param.nbytes, cudaMemcpyDeviceToHost));
                //CHECK(cudaMemcpy(aux_ps, d_ps3, param.nbytes, cudaMemcpyDeviceToHost));
                //fwrite(aux_pr, sizeof(float), param.nbxy, fp_pr);
                //fwrite(aux_ps, sizeof(float), param.nbxy, fp_ps);
                //// DELETE AFTER, only for testing

                //saveSnapshotIstep(it, d_ps3, param.modelNxBorder, param.modelNyBorder, "u3", shot);
                //float* h_aux = new float[param.nbxy];
                //char fname[32];
                //sprintf(fname, "snap/upgoing_%s_s%i_%i_%i_%i", "u3", shot, it, param.modelNyBorder, param.modelNxBorder);
                //FILE *snap = fopen(fname, "w");
                //CHECK(cudaMemcpy(h_aux, d_ps2, param.nbytes, cudaMemcpyDeviceToHost));
                //fwrite(h_aux, sizeof(float), param.nbxy, snap);
                //fflush(stdout);
                //fclose(snap);
                //sprintf(fname, "snap/downgoing_%s_s%i_%i_%i_%i", "u3", shot, it, param.modelNyBorder, param.modelNxBorder);
                //FILE *snap1 = fopen(fname, "w");
                //CHECK(cudaMemcpy(h_aux, d_pr2, param.nbytes, cudaMemcpyDeviceToHost));
                //fwrite(h_aux, sizeof(float), param.nbxy, snap);
                //fflush(stdout);
                //fclose(snap1);
                //sprintf(fname, "snap/product_%s_s%i_%i_%i_%i", "u3", shot, it, param.modelNyBorder, param.modelNxBorder);
                //FILE *snap2 = fopen(fname, "w");
                //CHECK(cudaMemcpy(h_aux, d_rtm, param.nbytes, cudaMemcpyDeviceToHost));
                //fwrite(h_aux, sizeof(float), param.nbxy, snap);
                //fflush(stdout);
                //fclose(snap2);


    // DELETE AFTER, only for testing
    //fclose(fp_ps);
    //fclose(fp_pr);
    //delete aux_pr, aux_ps;
    // DELETE AFTER, only for testing

    //// DELETE AFTER, only for testing
    //float *aux_ps = new float[param.nbxy];
    //float *aux_pr = new float[param.nbxy];
    //FILE *fp_ps = fopen("ShotWavefield", "w");
    //FILE *fp_pr = fopen("RecWavefield", "w");
    //// DELETE AFTER, only for testing
