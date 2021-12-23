#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "gpuconst.cuh"
#include "btree.cuh"

using namespace std;

#include "cudaKernels.cu"

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

//void test_kernel_applySourceArray(float dt, float *d_reflectivity, float *d_pField, float *d_vel, float *d_q)

void born(geometry param, velocity h_model, source h_wavelet, float *h_tapermask, seismicData h_seisData, sf_file Fonly_directWave, sf_file Fdata_directWave, sf_file Fdata, bool snaps)
{
    //cudaStream_t stream1;
    //cudaStreamCreate(&stream1);

    float dt2 = (h_wavelet.timeStep * h_wavelet.timeStep);
    float one_dx2 = float(1) / (param.modelDx * param.modelDx);
    float one_dy2 = float(1) / (param.modelDy * param.modelDy);
    //float dt2dx2 =
        //(h_wavelet.timeStep * h_wavelet.timeStep) / (param.modelDx * param.modelDx);   [> const for fd stencil <]
    size_t dbytes = param.nReceptors * h_wavelet.timeSamplesNt * sizeof(float);
    size_t tbytes = h_wavelet.timeSamplesNt * sizeof(float);

    //int bufferSize = min(param.nShots, 20);
    int bufferSize = min(param.nShots, 7);

    // Allocate memory on device
    printf("Allocate and copy memory on the device...\n");
    float *d_u1, *d_u2, *d_q1, *d_q2, *d_vp, *d_wavelet, *d_tapermask, *d_data, *d_directwave, *d_reflectivity;
    float *d_buffer, *d_lap;
    CHECK(cudaMalloc((void **)&d_u1, param.nbytes))       /* wavefield at t-2 */
    CHECK(cudaMalloc((void **)&d_u2, param.nbytes))       /* wavefield at t-1 */
    CHECK(cudaMalloc((void **)&d_q1, param.nbytes))       /* wavefield at t-2 */
    CHECK(cudaMalloc((void **)&d_q2, param.nbytes))       /* wavefield at t-1 */
    CHECK(cudaMalloc((void **)&d_vp, param.nbytes))       /* velocity model */
    CHECK(cudaMalloc((void **)&d_lap, param.nbytes))       /* velocity model */
    CHECK(cudaMalloc((void **)&d_wavelet, tbytes)); /* source term for each time step */
    CHECK(cudaMalloc((void **)&d_tapermask, param.nbytes));
    CHECK(cudaMalloc((void **)&d_reflectivity, param.nxy * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_data, dbytes));
    CHECK(cudaMalloc((void **)&d_directwave, dbytes));
    CHECK(cudaMalloc((void **)&d_buffer, dbytes * bufferSize));

    // Fill allocated memory with a value
    CHECK(cudaMemset(d_u1, 0, param.nbytes))
    CHECK(cudaMemset(d_u2, 0, param.nbytes))
    CHECK(cudaMemset(d_data, 0, dbytes))

    // Copy arrays from host to device
    CHECK(cudaMemcpy(d_vp, h_model.extVelField, param.nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tapermask, h_tapermask, param.nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_reflectivity, h_model.reflecitivy, param.nxy * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wavelet, h_wavelet.timeSeries, tbytes, cudaMemcpyHostToDevice));

    // Copy constants to device constant memory
    float coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpyToSymbol(c_coef, coef, 5 * sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_nx, &param.modelNxBorder, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_ny, &param.modelNyBorder, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nr, &param.nReceptors, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nxy, &param.nbxy, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nb, &param.taperBorder, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nt, &h_wavelet.timeSamplesNt, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_dt2, &dt2, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_one_dx2, &one_dx2, sizeof(float)));
    CHECK(cudaMemcpyToSymbol(c_one_dy2, &one_dy2, sizeof(float)));
    printf("\t%f MB\n", (4 * param.nbytes + tbytes)/1024/1024);
    printf("OK\n");

    // Print out specs of the main GPU
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("GPU0:\t%s\t%d.%d:\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    printf("\t%lu GB:\t total Global memory (gmem)\n", deviceProp.totalGlobalMem / 1024 / 1024 / 1000);
    printf("\t%lu MB:\t total Constant memory (cmem)\n", deviceProp.totalConstMem / 1024);
    printf("\t%lu MB:\t total Shared memory per block (smem)\n", deviceProp.sharedMemPerBlock / 1024);
    printf("\t%d:\t total threads per block\n", deviceProp.maxThreadsPerBlock);
    printf("\t%d:\t total registers per block\n", deviceProp.regsPerBlock);
    printf("\t%d:\t warp size\n", deviceProp.warpSize);
    printf("\t%d x %d x %d:\t max dims of block\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("\t%d x %d x %d:\t max dims of grid\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    CHECK(cudaSetDevice(0));

    // Print out CUDA domain partitioning info
    //printf("CUDA:\n");
    //printf("\t%i x %i\t:block dim\n", BDIMY, BDIMX);
    //printf("\t%i x %i\t:shared dim\n", SDIMY, SDIMX);
    //printf("CFL:\n");
    //printf("\t%f\n", _vp * h_wavelet.timeStep / dx);

    // Setup CUDA run
    dim3 block(BDIMX, BDIMY);
    dim3 grid((param.modelNxBorder + block.x - 1) / block.x, (param.modelNyBorder + block.y - 1) / block.y);

    //test_kernel_add_sourceArray(d_reflectivity, param, grid, block);


    int seismicSize = param.nReceptors * h_wavelet.timeSamplesNt;
    int step = 0;

    // MAIN LOOP
    for(int shot=0; shot<param.nShots; shot++){
        cerr<<"\nShot "<<shot<<" param.firstReceptorPos = "<<param.firstReceptorPos<<", param.srcPosX = "<<param.srcPosX<<", param.srcPosY = "<<param.srcPosY<<
            ", param.incShots = "<<param.incShots<<"\n"<<endl;

        CHECK(cudaMemset(d_u1, 0, param.nbytes))
        CHECK(cudaMemset(d_u2, 0, param.nbytes))
        CHECK(cudaMemset(d_q1, 0, param.nbytes))
        CHECK(cudaMemset(d_q2, 0, param.nbytes))


        float *d_u3, *d_q3;
        printf("Time loop...\n");
        for (int it = 0; it < h_wavelet.timeSamplesNt; it++)
        {
            taper_gpu<<<grid,block>>>(d_tapermask, d_u1);
            //testegoxtoso<<<grid,block>>>(d_u1);
            taper_gpu<<<grid,block>>>(d_tapermask, d_u2);
            //if(it == 0) saveSnapshotIstep(it, d_u1, param.modelNxBorder, param.modelNyBorder, "u1", shot);

            // These kernels are in the same stream so they will be executed one by one
            //kernel_2dfd<<<grid, block>>>(d_u1, d_u2, d_vp);
            kernel_2dfd_ver2<<<grid, block>>>(d_lap, d_u1, d_u2, d_vp);
            kernel_add_wavelet<<<grid, block>>>(d_u2, d_wavelet, it, param.srcPosX, param.srcPosY);

            taper_gpu<<<grid,block>>>(d_tapermask, d_q1);
            taper_gpu<<<grid,block>>>(d_tapermask, d_q2);

            // These kernels are in the same stream so they will be executed one by one
            kernel_2dfd<<<grid, block>>>(d_q1, d_q2, d_vp);
            //kernel_applySourceArray<<<grid, block>>>(h_wavelet.timeStep, d_reflectivity, d_u2, d_vp, d_q1);
            kernel_applySourceArray_ver2<<<grid, block>>>(h_wavelet.timeStep, d_reflectivity, d_lap, d_vp, d_q1);

            receptors<<<(param.nReceptors + 32) / 32, 32>>>(it, param.nReceptors, param.firstReceptorPos, d_q1, d_data);

            // Save snapshot every h_wavelet.snapStep iterations
            if ((it % h_wavelet.snapStep == 0) && snaps == true)
            {
                printf("%i/%i\n", it+1, h_wavelet.timeSamplesNt);
                saveSnapshotIstep(it, d_u1, param.modelNxBorder, param.modelNyBorder, "u1", shot);
                saveSnapshotIstep(it, d_q1, param.modelNxBorder, param.modelNyBorder, "q1", shot);
            }

            // Exchange time steps
            d_u3 = d_u1;
            d_u1 = d_u2;
            d_u2 = d_u3;

            d_q3 = d_q1;
            d_q1 = d_q2;
            d_q2 = d_q3;
        }

        //CHECK(cudaMemcpyAsync(h_seisData.seismogram, d_data, dbytes, cudaMemcpyDeviceToHost, stream1));
        //CHECK(cudaMemcpy(h_seisData.seismogram, d_data, dbytes, cudaMemcpyDeviceToHost));

        sf_warning("in_step=%d",step);
        if (step == bufferSize)
        {
            sf_warning("shot - bufferSize + 1 = %d",shot - bufferSize);
            CHECK(cudaMemcpy(&h_seisData.seismogram[(shot - bufferSize) * seismicSize], d_buffer, dbytes * bufferSize, cudaMemcpyDeviceToHost));
            step = 0;
        }
        CHECK(cudaMemcpy(&d_buffer[step * seismicSize], d_data, dbytes, cudaMemcpyDeviceToDevice));
        step += 1;



        param.firstReceptorPos += param.incRec;
        param.srcPosX += param.incShots;
    }


    sf_warning("out_step=%d",step);
    if(step < bufferSize)
    {
        CHECK(cudaMemcpy(&h_seisData.seismogram[(param.nShots - step) * seismicSize], d_buffer, dbytes * step, cudaMemcpyDeviceToHost));
    }
    else if (param.nShots == bufferSize)
    {
        CHECK(cudaMemcpy(h_seisData.seismogram, d_buffer, dbytes * param.nShots, cudaMemcpyDeviceToHost));
    }

    sf_floatwrite(h_seisData.seismogram, param.nReceptors * h_wavelet.timeSamplesNt * param.nShots, Fdata);

    printf("OK\n");

    CHECK(cudaGetLastError());


    CHECK(cudaFree(d_q1));
    CHECK(cudaFree(d_q2));
    CHECK(cudaFree(d_u1));
    CHECK(cudaFree(d_lap));
    CHECK(cudaFree(d_u2));
    CHECK(cudaFree(d_tapermask));
    CHECK(cudaFree(d_reflectivity));
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_directwave));
    CHECK(cudaFree(d_vp));
    CHECK(cudaFree(d_wavelet));
    CHECK(cudaFree(d_buffer));
    printf("OK saigo\n");
    CHECK(cudaDeviceReset());
    //cudaStreamDestroy(&stream1);
}
