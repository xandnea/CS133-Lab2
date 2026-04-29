// Header inclusions, if any...

#include <mpi.h>
#include <string.h>

#include "lib/gemm.h"
#include "lib/common.h"
// You can directly use aligned_alloc
// with lab2::aligned_alloc(...)

// Using declarations, if any...
#define BLOCK_SIZE 64
#define BI_SIZE 64
#define BJ_SIZE 1024
#define BK_SIZE 4

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // divide size kK into equal blocks for each thread to work on rowwise
  const int num_rows = kI / size;

  // allocate local buffers for processes to hold their factor rows of A, entirety of B, and the product rows of C
  float *a_local = (float *)lab2::aligned_alloc(64, num_rows * kK * sizeof(float)); // align to 64 bytes for better cache performance
  float *b_global = (float *)lab2::aligned_alloc(64, kK * kJ * sizeof(float));
  float *c_local = (float *)lab2::aligned_alloc(64, num_rows * kJ * sizeof(float));

  // zero out local C buffer to clear before accumulation
  memset(c_local, 0, num_rows * kJ * sizeof(float));

  // scatter rows of A to each thread 
  // sendbuf: a, sendcount: num_rows * kK, sendtype: MPI_FLOAT, recvbuf: a_local, recvtype: MPI_FLOAT, root: p0, comm: MPI_COMM_WORLD
  MPI_Scatter(a, num_rows * kK, MPI_FLOAT, a_local, num_rows * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  // on rank 0, copy the entirety of B into the global buffer to be broadcasted to all threads
  if (rank == 0) {
    memcpy(b_global, b, kK * kJ * sizeof(float));
    //printf("Rows per process: %d, num processes: %d\n", num_rows, size);
  }

  // broadcast entirety of B to all threads
  // sendbuf: b, sendcount: kK * kJ, sendtype: MPI_FLOAT, root: p0, comm: MPI_COMM_WORLD
  MPI_Bcast(b_global, kK * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // iterate blockwise through rows of A and C 
  for (int ii = 0; ii < num_rows; ii += BI_SIZE) {
    // iterate blockwise through columns of B and C
    for (int jj = 0; jj < kJ; jj += BJ_SIZE) {
      // iterate blockwise through col of A and rows of B
      for (int kk = 0; kk < kK; kk += BK_SIZE) {
        // iterate through ii block (rows of A and C)
        for (int i = ii; i < ii + BI_SIZE; i++) {

          // cache local pointer for row of A
          float* a_row = &a_local[i * kK];
          float* c_row = &c_local[i * kJ];

          // iterate through kk block (col of A and row of B)
          for (int j = jj; j < jj + BJ_SIZE; j+=8) {

            // cache registers for c values 
            float c0 = c_row[j];
            float c1 = c_row[j + 1];
            float c2 = c_row[j + 2];
            float c3 = c_row[j + 3];
            float c4 = c_row[j + 4];
            float c5 = c_row[j + 5];
            float c6 = c_row[j + 6];
            float c7 = c_row[j + 7];

            // iterate through column of B and C 
            for (int k = kk; k < kk + BK_SIZE; k++) {
              float a_ik = a_row[k];
              const float* b_row = &b_global[k * kJ + j];

              c0 += a_ik * b_row[0];
              c1 += a_ik * b_row[1];
              c2 += a_ik * b_row[2];
              c3 += a_ik * b_row[3];
              c4 += a_ik * b_row[4];
              c5 += a_ik * b_row[5];
              c6 += a_ik * b_row[6];
              c7 += a_ik * b_row[7];
            }
            c_row[j] = c0;
            c_row[j + 1] = c1;
            c_row[j + 2] = c2;
            c_row[j + 3] = c3;
            c_row[j + 4] = c4;
            c_row[j + 5] = c5;
            c_row[j + 6] = c6;
            c_row[j + 7] = c7;
          }
        }
      }
    }
  }

  // gather the rows of C calculated by each process back to the root process
  // sendbuf: c_local, sendcount: num_rows * kJ, sendtype: MPI_FLOAT, recvcount: num_rows * kJ, recvtype: MPI_FLOAT, root: p0, comm: MPI_COMM_WORLD
  MPI_Gather(c_local, num_rows * kJ, MPI_FLOAT, c, num_rows * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  // free individual allocated memory per process
  free(a_local), free(b_global), free(c_local);
}