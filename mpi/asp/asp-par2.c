#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mpi.h"

#define MAX_DISTANCE 256
// #define VERBOSE

/* malloc and initialize the table with some random distances       */
/* we never use srand() so rand() will always use the same seed     */
/* and will hence yields reproducible results (for timing purposes) */


// random initialization of the matrix 
// to be used only for testing purposes 
void init_tab(int n, int *mptr, int ***tabptr, int oriented)
{
	int **tab;
	int i, j, m=n*n;

	tab = (int **)malloc(n * sizeof(int *));
	if (tab == (int **)0) {
		fprintf(stderr,"cannot malloc distance table\n");
		exit (42);
	}

	for (i = 0; i < n; i++) {
		tab[i]    = (int *)malloc(n * sizeof(int));
		if (tab[i] == (int *)0) {
			fprintf(stderr,"cannot malloc distance table\n");
			exit (42);
		}
		tab[i][i]=0;
		for (j = 0; j < i; j++) {
			tab[i][j] = 1+(int)((double)MAX_DISTANCE*rand()/(RAND_MAX+1.0));
			if (oriented) 
				tab[j][i] = 1+(int)((double)MAX_DISTANCE*rand()/(RAND_MAX+1.0));
			else 
				tab[j][i] = tab[i][j];
			if (tab[i][j]==MAX_DISTANCE) m--;
			if (tab[j][i]==MAX_DISTANCE) m--; 
		}
	}
	*tabptr = tab;
	*mptr = m;
}


// reading the list of edges from a file and constructing an adjacency matrix 
// note that it is not mandatory for the graph to be stored as an adjacency matrix - 
// other data structures are allowed, and should be chosen depending on the chosen 
// implementation for the APSP algorithm. 

// The file has the following format: 
// first line: [number of vertices] [number of edges] [oriented(0/1)]
// following [number of edges lines]: [source_node] [destination_node] [weight]
int read_tab(char *INPUTFILE, int *nptr, int *mptr, int ***tabptr, int *optr) 
/*
INPUTFILE = name of the graph file 
nptr = number of vertices, to be read from the file 
mptr = number of edges, to be read from the file 
tabptr = the adjancecy matrix for the graph
optr = returns 1 when the graph is oriented, and 0 otherwise. 

returns: the number of edges that are "incorrect" in the file. That is, in case 
the graph is not oriented, but there are different entries for symmetrical pairs 
of edges, the second such edge is ignored, yet counted for statistics reasons.
E.g.: 

1 5 20 
5 1 50 

-> If the graph is oriented, these entries are both copied to the adjancency matrix:
A[1][5] = 20, A[5][1] = 50
-> If the graph is NOT oriented, the first entry is copied for both pairs, and the second 
one is discarded: A[1][5] = A[5][1] = 20 ; this is a case for an incorrect edge. 

NOTE: the scanning of the file is depenedent on the implementation and the chosen 
data structure for the application. However, it has to be the same for both the sequential 
and the parallel implementations. For the parallel implementation, the file is read by a 
single node, and then distributed to the rest of the participating nodes. 
File reading and graph constructions should not be considered for any timing results. 
*/
{
	int **tab;
	int i,j,n,m;
	int source, destination, weight;
	FILE* fp; 
	int bad_edges=0, oriented=0;

	fp=fopen(INPUTFILE, "r");
  if(fp == NULL){
    fprintf(stderr,"Error opening the file\n");
    exit(1);
  }
	fscanf(fp, "%d %d %d \n", &n, &m, &oriented);
  

        tab = (int **)malloc(n * sizeof(int *));
        if (tab == (int **)0) {
                fprintf(stderr,"cannot malloc distance table\n");
                exit (42);
        }

        for (i = 0; i < n; i++) {
                tab[i]    = (int *)malloc(n * sizeof(int));
		if (tab[i] == (int *)0) {
                        fprintf(stderr,"cannot malloc distance table\n");
                        exit (42);
                }
		
		for (j = 0; j < n; j++) {
                        tab[i][j] = (i == j) ? 0 : MAX_DISTANCE;
                }
	}

	while (!feof(fp)) {
		fscanf(fp, "%d %d %d \n", &source, &destination, &weight);
		if (!oriented) {
			if (tab[source-1][destination-1] < MAX_DISTANCE) 
				bad_edges++;
			else {
				tab[source-1][destination-1]=weight;
				tab[destination-1][source-1]=weight;
			}
		}
		else 
			tab[source-1][destination-1]=weight;
		
	}
	fclose(fp);
#ifdef VERBOSE 
	for (i=0; i<n; i++) {
		for (j=0; j<n; j++) 
			printf("%5d", tab[i][j]);
		printf("\n");
	}
#endif 

	*tabptr=tab;
	*nptr=n;
	*mptr=m;
	*optr=oriented;
	return bad_edges; 
}

void init_next(int n, int ***nextptr){
  int **next;
  int i, j;

  next = (int**) malloc(n * sizeof(int*));
  if(next == (int **)0){
    fprintf(stderr,"cannot malloc next table\n");
    exit(42);
  }

  for(i=0;i<n;i++){
    next[i] = (int *)malloc(n * sizeof(int));
    if (next[i] == (int*)0){
      fprintf(stderr, "cannot malloc next table\n");
      exit(42);
    }
    for(j=0;j<n;j++){
      next[i][j] = j;
    }
  } 

  *nextptr = next;
}

void free_tab(int **tab, int n)
{
	int i;
    
	for (i = 0; i < n; i++) {
		free(tab[i]);
	}
	free(tab);
}


void print_tab(int **tab, int n)
{
	int i, j;

	for(i=0; i<n; i++) {
		for(j=0; j<n; j++) {
			printf("%2d ", tab[i][j]);
		}
		printf("\n");
	}
}

void print_rows(int **rows, int n, int m){
  int i,j;

  for(i=0;i<n;i++){
    for(j=0;j<m;j++){
      printf("%2d ", rows[i][j]);
    }
    printf("\n");
  }
}

void do_asp(int **rows, int n, int lb, int ub, int p, int id, int **next_rows){
	int i, j, k, tmp, proc;
  int *rowK;
  int **tab;
  MPI_Status status;

  tab = malloc(n * sizeof(int*));
  for(i=0;i<n;i++){
    tab[i] = malloc(n * sizeof(int));
    for(j=0;j<n;j++){
      tab[i][j] = -1;
    }
  }


	for (k = 0; k < n; k++) {
    rowK = (int*)(malloc((n+1) * sizeof(int)));
    if(k>=lb && k <= ub){ /*I already have the row*/
      //Store k in the row you're sending*/
      rowK[0] = k;
      for(i=1;i<n+1;i++){
        rowK[i] = rows[k-lb][i-1];
        tab[k][i-1] = rowK[i];
      }
      for(proc = 0;proc < p;proc++){ /*send the row to all other nodes */
        if(proc != id){
          MPI_Send(&rowK[0], n+1, MPI_INT, proc, 0, MPI_COMM_WORLD);
       //   printf("I'm %d and I've just sent row %d\n", id, k);
        }
      }
    }else{ 
      if(tab[k][0] == -1){/*I don't yet have this row */
        do{
          MPI_Recv(&rowK[0], n+1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
          for(i=0;i<n;i++){
            tab[rowK[0]][i] = rowK[i+1];
            //printf("I'm %d and just filled row %d. First value is %d\n", id, rowK[0], rowK[1]);
          }
        }while (rowK[0] != k);
      }
    }
    for (i = 0; i <= ub-lb; i++) { //ub-lb == rows_to_process-1
      for (j = 0; j < n ; j++) {
        tmp = rows[i][k] + tab[k][j];
     //   printf("I'm %d. tmp = %d. tab[%d][%d] = %d\n", id, tmp, k, j, tab[k][j]);
        if (tmp < rows[i][j]) {
          rows[i][j] = tmp;
          next_rows[i][j] = next_rows[i][k];
        }
      }
		}
  free(rowK);
  //MPI_Barrier(MPI_COMM_WORLD);
	}
  free_tab(tab, n);
}

int get_distance(int x, int y, int** tab){
  return(tab[x][y]);
}

int get_diameter(int **tab, int n){
  int i, j, max=0;

  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      if(i!=j){
        if(tab[i][j] > max){
          max = tab[i][j];
        }
      }
    }
  }
  return max;
}


/*determines and prints the route which corresponds to the shortest path determined by asp*/
void get_path(int city_1, int city_2, int **next, int **tab, int n){
  int a,b;
 
  if(city_1<0||city_2<0||city_1>n||city_2>n){
    fprintf(stderr,"One or both cities you have specified do not exist! City must be larger than zero and smaller than N\n");
    exit(42);
  }

  a = city_1;
  b = city_2;

  printf("The shortest route from city %d to city %d is: \n", city_1, city_2);
  do{
    printf("city %d to city %d (%d km) \n", a, next[a][b], tab[a][next[a][b]]);
    a = next[a][b];
  }while(a != city_2);
  printf("The total distance between city %d and city %d is %d km\n", city_1, city_2, get_distance(city_1, city_2, tab));
}



void usage() {
	printf ("Run the asp program with the following parameters. \n");
	printf (" -read filename :: reads the graph from a file.\n");
	printf (" -random N 0/1 :: generates a NxN graph, randomly. \n");
	printf ("               :: if 1, the graph is oriented, otherwise it is not oriented.\n");
  printf (" -diameter :: returns the largest distance between any two cities in the network.\n");
  printf (" -get city_1 city_2 :: returns the shortest route between two cities.\n");
  return ;
}


/******************** Main program *************************/

int main ( int argc, char *argv[] ) {
  int id;
  int ierr;
  int p=3;
  int random=0;
  int get=0;
  int diameter=0;
  int city_1, city_2;
  double wtime;
  int n,m, bad_edges=0, oriented=0, lb, ub, i, j, k;
  int **tab;
  int **rows;
  int **next;
  int **next_rows;
  int **buf;
  int print = 0;
  char FILENAME[100];
  int rows_to_process;
  int last_node_rows;
  MPI_Status *status;

  /*MPI stuff*/

  /*Initialize MPI.*/
  ierr = MPI_Init ( &argc, &argv );
        if(ierr != MPI_SUCCESS) {
                perror("Error with initializing MPI");
                exit(1);
        }

 /*Get the number of processes.*/
  ierr = MPI_Comm_size ( MPI_COMM_WORLD, &p );
 /*Get the individual process ID.*/
  ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &id );
 /*Process 0 reads data + prints welcome*/
  if (id==0) {
        usage();
  }

  n = 0;

	for(i=1; i<argc; i++) {
		if(!strcmp(argv[i], "-print")) {
			print = 1;
      continue;
		} 
		if (!strcmp(argv[i], "-read")) {
			strcpy(FILENAME, argv[i+1]); i++; 
      continue;
		}
		if (!strcmp(argv[i], "-random")) {
			n = atoi(argv[i+1]); 
			oriented = atoi(argv[i+2]); 
      i+=2;	
      random = 1;
      continue;
		}	
    if (!strcmp(argv[i], "-get")){
      city_1 = atoi(argv[i+1]);
      city_2 = atoi(argv[i+2]);
      get = 1;
      if(city_1 < 0 || city_2 < 0){
        usage();
        exit(1);
      }

      i+=2;
      continue;
    }
    if (!strcmp(argv[i], "-diameter")){
      diameter = 1;
      continue;
    }
	}



  if ( id == 0 ) 
  {
    wtime = MPI_Wtime ( );

    printf ( "\n" );
    printf ( "HELLO_MPI - Master process:\n" );
    printf ( "  C/MPI version\n" );
    printf ( "  An MPI example program.\n" );
    printf ( "\n" );
    printf ( "  The number of processes is %d.\n", p );
    printf ( "\n" );

    /* generate a random data set, or read in the file*/
    if (random) {
      printf("calling init_tab with n = %d\n", n);
      init_tab(n,&m,&tab,oriented); // last one = oriented or not ... 
    }else{
      bad_edges = read_tab(FILENAME, &n, &m, &tab, &oriented); 
    }

    init_next(n, &next);
  }

  
  /*Broadcast the value of n to all nodes*/
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);


  
  /*determine the upper and lower bound for each process*/
  //rows_to_process = ((float)(n)/(float)(p) + 0.5f);
  rows_to_process = (n + (p-1)) / p;
  lb = id*rows_to_process;
  ub = lb+rows_to_process-1;
  if(ub>(n-1)){
    ub = n-1;
    rows_to_process = n - ((p-1) * rows_to_process);
  }

  //printf("I'm %d, lb = %d, ub = %d, rtp = %d\n", id, lb, ub, rows_to_process);


  rows = malloc(rows_to_process * sizeof(int *));
  if(rows == NULL){
    fprintf(stderr,"Error allocating rows \n");
    exit(1);
  }
  
  next_rows = malloc(rows_to_process * sizeof(int *));
  if(next_rows == NULL){
    fprintf(stderr,"Error allocating next_rows \n");
    exit(1);
  }


  for(i=0;i<rows_to_process;i++){
    rows[i] = malloc(n * sizeof(int));
    if(rows[i] == NULL){
      fprintf(stderr,"Error allocating next_rows \n");
      exit(1);
    }
  }
  
  for(i=0;i<rows_to_process;i++){
    next_rows[i] = malloc(n * sizeof(int));
    if(next_rows[i] == NULL){
      fprintf(stderr,"Error allocating next_rows \n");
      exit(1);
    }
    for(j=0;j<n;j++){
      next_rows[i][j] = j;
    }
  }

  /*distribute data*/

    
  if(id==0){
    /*fill own rows with data from tab*/
    for(i=0;i<rows_to_process;i++){
      for(j=0;j < n ;j++){
        rows[i][j] = tab[i][j];
      }
    }
    /*send rows to other nodes, except the last, as this node may have a different number of rows to process!*/
    for(i=1;i<p-1;i++){
      buf = malloc(rows_to_process * sizeof(int *));
      for(j=0;j<rows_to_process;j++){
        buf[j] = malloc(n * sizeof(int));
        for(k=0;k<n;k++){
          buf[j][k] = tab[rows_to_process * i + j][k];
        }
      }
      /*send data per row to responsible process*/
      for(j=0;j<rows_to_process;j++){
        MPI_Send(&buf[j][0],n,MPI_INT,i,0,MPI_COMM_WORLD);
      }
      for(j=0;j< rows_to_process ;j++){
        free(buf[j]);
      }
      free(buf);
    }
    /*send rows to last node*/
    last_node_rows = n - ((p-1) * rows_to_process);
    buf = malloc(last_node_rows * sizeof(int *));
    for(j=0;j<last_node_rows;j++){
      buf[j] = malloc(n * sizeof(int));
      for(k=0;k<n;k++){
        buf[j][k] = tab[rows_to_process * (p-1) + j][k];
      }
    }
    /*send data per row to last process*/
    for(j=0;j<last_node_rows;j++){
      MPI_Send(&buf[j][0],n,MPI_INT,p-1,0,MPI_COMM_WORLD);
    }
    for(j=0;j< last_node_rows ;j++){
      free(buf[j]);
    }
    free(buf);
  }else{
    for(i=0;i<rows_to_process;i++){
      MPI_Recv(&rows[i][0],n,MPI_INT,0,0,MPI_COMM_WORLD,status);
    }
      
  }
  

  do_asp(rows,n,lb,ub,p,id, next_rows);

  /*send back computed data to master node*/
  if(id==0){
    /*copy data from rows to tab*/
    for(i=0;i<rows_to_process;i++){
      for(j=0;j<n;j++){
        tab[i][j] = rows[i][j];
        next[i][j] = next_rows[i][j];
      }
    }
    /*receive computed data from other nodes and store in tab*/
    for(i=1;i<p-1;i++){
      for(j=0;j<rows_to_process;j++){
        MPI_Recv(&tab[j + rows_to_process * i][0], n, MPI_INT,i,0,MPI_COMM_WORLD,status);
        MPI_Recv(&next[j + rows_to_process * i][0], n, MPI_INT,i,0,MPI_COMM_WORLD,status);
      }
    }
    for(j=0;j<last_node_rows;j++){
      MPI_Recv(&tab[j + rows_to_process * i][0], n, MPI_INT,i,0,MPI_COMM_WORLD,status);
      MPI_Recv(&next[j + rows_to_process * i][0], n, MPI_INT,i,0,MPI_COMM_WORLD,status);
    }
  }else{
    /* send data to process 0*/
    for(i=0;i<rows_to_process;i++){
      MPI_Send(&rows[i][0], n,MPI_INT,0,0,MPI_COMM_WORLD);
      MPI_Send(&next_rows[i][0], n,MPI_INT,0,0,MPI_COMM_WORLD);
    }
  }
  for(i=0;i<rows_to_process;i++){
    free(rows[i]);
  }
  free(rows);




  if(id != 0){
    MPI_Finalize();
    return 0;
  }

/*Process 0 says goodbye.*/

  if ( id == 0 )
  {
    printf ( "\n" );
    printf ( "HELLO_MPI - Master process:\n" );
    printf ( "  Normal end of execution: 'Goodbye, world!'\n" );

    wtime = MPI_Wtime ( ) - wtime;
    printf ( "\n" );
    printf ( "  Elapsed wall clock time = %f seconds.\n", wtime );

  }
/*
  Shut down MPI.
*/
  ierr = MPI_Finalize ( );
  if(get){
    get_path(city_1, city_2, next, tab, n);
  }
  if(print){
    print_tab(tab, n);
  }
  if(diameter){
    diameter = get_diameter(tab, n);
    printf("The diameter of the network is: %d\n", diameter);
  }

  return 0;
}
