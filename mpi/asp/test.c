#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv){
  int rank, size, msg;
  MPI_Status status;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Hello world from process %d of %d\n", rank, size);
  MPI_Send(&rank, 1, MPI_INT, (rank+1)%size,0, MPI_COMM_WORLD);
  MPI_Recv(&msg, 1, MPI_INT, (rank-1)%size,0, MPI_COMM_WORLD, &status); 
  printf("I (process %d) received a message from process %d\n", rank, msg);
  MPI_Finalize();
  return 0;
}

