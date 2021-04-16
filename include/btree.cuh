#ifndef BTREE_CUH
#define BTREE_CUH

#include <rsf.hh>

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

// Check error codes for CUDA functions
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

typedef struct{
    int nShots;
    int srcPosX;
    int srcPosY;
    int firstReceptorPos;
    int nReceptors;
    int lastReceptorPos;
    int incShots;
    int incRec;
    int modelNx;
    int modelNy;
    int modelNxBorder;
    int modelNyBorder;
    float modelDx;
    float modelDy;
    int taperBorder;
    // Auxiliaries
    size_t nxy;
    size_t nbxy;
    size_t nbytes;
} geometry;

typedef struct{
    float *velField;
    float *extVelField;
    float *firstLayerVelField;
    float *reflecitivy;
    float maxVel;
} velocity;

typedef struct{
    float *seismogram;
    float timeStep;
    int timeSamplesNt;
    float *directWaveOnly;
} seismicData;

typedef struct{
    float totalTime;
    float timeStep;
    int timeSamplesNt;
    int snapStep;
    float *timeSeries;
} source;


__global__ void receptors(int it, int nr, int gxbeg, float *d_u1, float *d_data);
__global__ void kernel_add_wavelet(float *d_u, float *d_wavelet, int it, int jsrc, int isrc);
__global__ void kernel_add_seismicdata(int it, int nr, int gxbeg, float *d_u1, float *d_seisdata);
__global__ void kernel_image_condition(float *d_u, float *d_q, float *d_rtm);
__global__ void kernel_add_sourceArray(float *d_u, float *d_sourceArray);
__global__ void kernel_applySourceArray(float dt, float *d_reflectivity, float *d_pField, float *d_vel, float *d_q);
__device__ void set_halo(float *global, float shared[][SDIMX], int tx, int ty, int sx, int sy, int gx, int gy, int nx, int ny);
__global__ void kernel_2dfd(float *d_u1, float *d_u2, float *d_vp);
__global__ void kernel_lap_filter(float *d_rtm);
void saveSnapshotIstep(int it, float *data, int nx, int ny, const char *tag, int shot);
__global__ void taper_gpu (float *d_tapermask, float *campo);
__global__ void frequency (int nFreq, float df, float *DFTconstTerm);
__global__ void nucleoDFT(int nFreq, float *DFTconstTerm, float *RealKernel, float *ImagKernel);

__global__ void freqKernelImg(int it, int iw, int nt, float *DFTconstTerm, float *RealKernel, float *ImagKernel,
        float *sourceRealPart, float *sourceImagPart, float *receptRealPart, float *receptImagPart,
        float *d_ps1, float *d_pr1, float *d_rtm);

void dummyVelField(int nxb, int nyb, int nb, float *h_vpe, float *h_dvpe);
void expand(int nb, int nyb, int nxb, int nz, int nx, float *a, float *b);
void abc_coef (int nb, float *abc);
void taper (int nx, int ny, int nb, float *abc, float *campo);
sf_file createFile3D (const char *name, int dimensions[3], float spacings[3], int origins[3]);
geometry getParameters(sf_file FvelModel);
velocity getVelFields(sf_file FvelModel, sf_file Freflectivity, geometry param);
float* tapermask(geometry param);
seismicData allocHostSeisData(geometry param, int nt);
source fillSrc(geometry param, velocity h_model);
void test_getParameters (geometry param, source wavelet);
void born(geometry param, velocity h_model, source h_wavelet, float *h_tapermask, seismicData h_seisData, sf_file Fonly_directWave, sf_file Fdata_directWave, sf_file Fdata, bool snaps);

void modeling(geometry param, velocity h_model, source h_wavelet, float *h_tapermask, seismicData h_seisData, sf_file Fonly_directWave, sf_file Fdata_directWave, sf_file Fdata, int snaps);

void rtm(geometry param, velocity h_model, float *h_wavelet, float *h_tapermask, seismicData h_seisData, sf_file Fdata);
void frtm(geometry param, velocity h_model, float *h_wavelet, float *h_tapermask, seismicData h_seisData, sf_file Fdata);


__host__ void hostfreqKernelImg(int nb, int nt, int nzb, int nxb, int it, int iw, float *RealKernel, float *ImagKernel,
        float *sourceRealPart, float *sourceImagPart, float *receptRealPart, float *receptImagPart,
        float *h_ps1, float *h_pr1, float *h_rtm);

void test_getParameters (geometry param, seismicData h_seisData);


float* fillSrc(geometry param, velocity h_model, seismicData h_seisData);
geometry getParameters(sf_file FvelModel, sf_file Fshots);
seismicData allocHostSeisData(geometry param, sf_file Fshots);

velocity getVelFields(sf_file FvelModel, geometry param);

#endif
