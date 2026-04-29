// Header inclusions, if any...

#include <mpi.h>
#include <string.h>

#include "lib/gemm.h"
#include "lib/common.h"
// You can directly use aligned_alloc
// with lab2::aligned_alloc(...)

// Using declarations, if any...
#define BLOCK_SIZE 64

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
  }

  // broadcast entirety of B to all threads
  // sendbuf: b, sendcount: kK * kJ, sendtype: MPI_FLOAT, root: p0, comm: MPI_COMM_WORLD
  MPI_Bcast(b_global, kK * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // iterate blockwise through rows of A and C 
  for (int ii = 0; ii < num_rows; ii += 64) {
    // iterate blockwise through columns of B and C
    for (int jj = 0; jj < kJ; jj += 1024) {
      // iterate blockwise through col of A and rows of B
      for (int kk = 0; kk < kK; kk += 4) {
        // iterate through ii block (rows of A and C)
        for (int i = ii; i < ii + 64; i++) {
          // iterate through kk block (col of A and row of B)
          for (int j = jj; j < jj + 1024; j++) {

            // cache local pointers for element of C, row of A, and column of B
            float* c_ij = &c_local[i * kJ + j];
            float* a_row = &a_local[i * kK];
            const float* b_col = &b_global[j];

            // iterate through column of B and C 
            for (int k = kk; k < kk + 4; k++) {
              *c_ij += a_row[k] * b_col[k * kJ];
            }
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
