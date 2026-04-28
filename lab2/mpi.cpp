// Header inclusions, if any...

#include <mpi.h>

#include "lib/gemm.h"
#include "lib/common.h"
// You can directly use aligned_alloc
// with lab2::aligned_alloc(...)

// Using declarations, if any...
#define BLOCK_SIZE 128

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // divide size kK into equal blocks for each thread to work on rowwise
  const int num_rows = kI / size;

  // allocate local buffers for processes to hold their factor rows of A, entirety of B, and the product rows of C
  float *a_local = (float *)lab2::aligned_alloc(num_rows * kK * sizeof(float));
  float *b_global = (float *)lab2::aligned_alloc(kK * kJ * sizeof(float));
  float *c_local = (float *)lab2::aligned_alloc(num_rows * kJ * sizeof(float));

  // zero out local C buffer to clear before accumulation
  memset(c_local, 0, num_rows * kJ * sizeof(float));

  // scatter rows of A to each thread 
  // sendbuf: a, sendcount: num_rows * kK, sendtype: MPI_FLOAT, recvbuf: a_local, recvtype: MPI_FLOAT, root: p0, comm: MPI_COMM_WORLD
  MPI_Scatter(a, num_rows * kK, MPI_FLOAT, a_local, num_rows * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  // broadcast entirety of B to all threads
  // sendbuf: b, sendcount: kK * kJ, sendtype: MPI_FLOAT, root: p0, comm: MPI_COMM_WORLD
  MPI_Bcast(b, kK * kJ, MPI_FLOAT, 0, MPI_COMM_WORLD);

  for (int c_i = 0; c_i < num_rows; c_i += 64) {
    for (int c_j = 0; c_j < kJ; c_j += 1024) {
      // in a thread: working on block (c_i, c_j) of C

      // iterate horizontally through A blocks & vertically through B blocks 
      // block offset is calculated to add to the k dimensions of A and B
      for (int block_iter = 0; block_iter < num_blocks; block_iter++) {
        int block_offset = block_iter * BLOCK_SIZE;

        // iterate through the block of A and B, and update the block of C
        for (int i = 0; i < BLOCK_SIZE; i += 2) { // unroll the i loop by 2
          for (int k = 0; k < BLOCK_SIZE; k++) {

            // hoist two values of A out of the innermost loop since it doesn't change across j
            const float a_i0_k = a_local[(c_i + i) * kK + (k + block_offset)];
            const float a_i1_k = a_local[(c_i + i + 1) * kK + (k + block_offset)];

            // utilize row pointers for each c row being calculated at column j
            float* c_row0 = &c_local[(c_i + i) * kJ + c_j];
            float* c_row1 = &c_local[(c_i + i + 1) * kJ + c_j];
            const float* b_row = &b_global[(k + block_offset) * kJ + c_j];

            for (int j = 0; j < BLOCK_SIZE; j++) {
              // one load of b_row[j] feeds two updates
              float b_val = b_row[j];
              c_row0[j] += a_i0_k * b_val;
              c_row1[j] += a_i1_k * b_val;
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
