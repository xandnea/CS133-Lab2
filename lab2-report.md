# CS 133 Lab 2 Report: MPI-Based Parallel GEMM
**Name:** Alexander Neary  
**Date:** April 2026  

---

## 1. Data and Computation Partitioning

### 1.1 Data Partitioning
The implementation uses a 1D row-wise decomposition for Matrix A and complete replication for Matrix B.
- **Matrix A:** Processor 0 partitions the rows of A into different and equal chunks based on the number of available processors ($num\_rows = kI / size$).
- **Matrix B:** Since every processor needs to access all columns of B to compute the rows of C (using the rows of A they have), the entirety of Matrix B gets sent to every processor.
- **Matrix C:** Similar to Matrix A, each processor only needs to store its results in the `num_rows` that it's assigned to. The results are combined after the computation of each group of rows per processor is complete.

### 1.2 Computation Partitioning
The computation is parallelized by assigning each MPI rank a specific subset of rows from Matrix A to calculate the same product rows in Matrix C. Each process independently executes a blocked $i \rightarrow j \rightarrow k$ loop structure to maximize cache reuse and spatial locality.

### 1.3 Communication Strategy
- **Distribution:** `MPI_Scatter()` is used to send the row chunks of A out to every processor. Similarly, `MPI_Bcast()` is used to send the entire Matrix B to each processor for processing.
- **Collection:** `MPI_Gather()` is utilized once all computation across processors 0-3 is completed to recombine the rows of C into a full result Matrix C back in processor 0.

---

## 2. Analysis of Communication APIs

### 2.1 Theoretical Analysis
- **Blocking (MPI_Send/Recv):** Simplest to implement, but can lead to idle time waiting for handshakes to complete, increasing overhead or, in the worst case, deadlocks.
- **Non-Blocking (MPI_Isend/Irecv):** Slightly more complex; allows for communication and computation to overlap, meaning a new block of B could theoretically be fetched while computing the current one.
- **Collective (Scatter/Bcast/Gather):** These were the ideal MPI commands for matrix multiplication as they generalize the message passing to $p0 \rightarrow all$ and $all \rightarrow p0$, which works perfectly for row-wise decomposition.

### 2.2 Experimental Choice
In testing, the choice was between looping through each process and manually using `MPI_Send` or utilizing collective commands to have process 0 send everything at once. Utilizing collective commands reduced the bottleneck at the root process, resulting in higher performance.

---

## 3. Performance Evaluation

### 3.1 Performance Improvements
The following optimizations were performed iteratively to reach the final performance:

- **MPI Infrastructure and 1D Blocking:** Established the distributed-memory model using `MPI_Scatter()`, `MPI_Bcast()`, and `MPI_Gather()`.
- **Better Blocking and $i \rightarrow k \rightarrow j$ Loop Order:** Matched the Lab 1 $i, k, j$ order to improve memory access performance (row-wise instead of column-wise).
- **Updated to $i \rightarrow j \rightarrow k$ Order with Uneven Blocking:** This order allowed for better cache locality with pointers/registers. Adjusted block sizes to be unequal, prioritizing rows of B (1024).
- **Added Row-Wise Loop Unrolling:** Initially unrolled by a factor of 2, then increased to a factor of 4 to access and perform calculations on more values while they are in cache.
- **K-Block Tuning:** Increased `BK_SIZE` to 8 from 4, which better utilized the L1/L2 cache during inner product calculations.

### 3.2 Result Table
The following results were obtained on the **m5.2xlarge** AWS instance:

| Problem Size |  GFlops  | Time (s) | Performance Range |
|--------------|----------|----------|-------------------|
| $1024^3$     | 68.7963  | 0.03121  | C                 |
| $2048^3$     | 88.4427  | 0.1942   | B                 |
| $4096^3$     | 137.0760 | 1.0026   | **A**             |
|--------------------------------------------------------|

**Performance Range achieved: A**

---

## 4. Scalability
Testing was performed on the $4096^3$ problem size by varying the number of processors ($np$):

- **$np=1$:** ~41 GFlops
- **$np=2$:** ~74 GFlops
- **$np=4$:** ~137 GFlops
- **$np=8$:** ~128 GFlops
- **$np=16$:** ~87 GFlops
- **$np=32$:** ~60 GFlops

**Discussion of Non-Linearity:** The program scales almost linearly from 1 to 4 processors, but performance drops off beyond that point. This is likely due to the high cost of broadcasting Matrix B across significantly more cores, where communication overhead begins to outweigh computational gains.

---

## 5. MPI vs OpenMP

### 5.1 Programming Effort
MPI required more effort due to the necessity of manual memory allocation (using `aligned_alloc`) and the direct management of data distribution and collection. Coordinating results across processes was more challenging than the shared-memory model of OpenMP.

### 5.2 Performance Difference
- **OpenMP (Lab 1):** ~72 GFlops
- **MPI (Lab 2):** ~137 GFlops

**Analysis:** MPI achieved a higher peak performance because each process operates in its own private address space. This minimizes the risk of "False Sharing," where different processors fight over the same cache line—a constant concern in OpenMP. While MPI has higher message-passing overhead, its performance on larger problem sizes is superior for this architecture.