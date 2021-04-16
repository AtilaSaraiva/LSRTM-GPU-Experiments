#include "btree.cuh"
#include <stdio.h>

// Save snapshot as a binary, filename snap/snap_tag_it_ny_nx
void saveSnapshotIstep(int it, float *data, int nx, int ny, const char *tag, int shot)
{
    /*
    it      :timestep id
    data    :pointer to an array in device memory
    nx, ny  :model dimensions
    tag     :user-defined file identifier
    */

    // Array to store wavefield
    unsigned int isize = nx * ny * sizeof(float);
    float *iwave = (float *)malloc(isize);
    CHECK(cudaMemcpy(iwave, data, isize, cudaMemcpyDeviceToHost));

    char fname[32];
    sprintf(fname, "snap/snap_%s_s%i_%i_%i_%i", tag, shot, it, ny, nx);

    FILE *fp_snap = fopen(fname, "w");

    fwrite(iwave, sizeof(float), nx * ny, fp_snap);
    printf("\tSave...%s: nx = %i ny = %i it = %i tag = %s\n", fname, nx, ny, it, tag);
    fflush(stdout);
    fclose(fp_snap);

    free(iwave);
    return;
}
