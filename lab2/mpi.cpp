// Header inclusions, if any...

#include <mpi.h>

#include "lib/gemm.h"
#include "lib/common.h"
// You can directly use aligned_alloc
// with lab2::aligned_alloc(...)

// Using declarations, if any...
#define NUM_THREADS 8
#define BLOCK_SIZE 128

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ], float c[kI][kJ]) {
  const int num_blocks = kK / BLOCK_SIZE;

  MPI_Scatter() // scatter the rows of C to different threads, so each thread is responsible for calculating a different block of C
  for (int c_i = 0; c_i < (c_i + 1) * (kI / NUM_THREADS); c_i += 64) {
    MPI_Bcast() // broadcast the block of A needed for this thread to all threads
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
            const float a_i0_k = a[c_i + i][k + block_offset];
            const float a_i1_k = a[c_i + i + 1][k + block_offset];

            // utilize row pointers for each c row being calculated at column j
            float* c_row0 = &c[c_i + i][c_j];
            float* c_row1 = &c[c_i + i + 1][c_j];
            const float* b_row = &b[k + block_offset][c_j];

            for (int j = 0; j < BLOCK_SIZE; j++) {
              MPI_Gather() // gather the block of B needed for this thread from all threads, so each thread has the necessary block of B to calculate its block of C
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
}
