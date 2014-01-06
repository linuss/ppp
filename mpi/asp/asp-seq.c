/*
 * asp.c:
 * 	All-pairs shortest path implementation based on Floyd's
 * 	algorithms.
 *
 *      Sequential version.
 *
 * Written by Rob van Nieuwpoort, 6-oct-2003
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/******************** ASP stuff *************************/


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
	fscanf(fp, "%d %d %d \n", &n, &m, &oriented);
//#ifdef VERBOSE
	printf("%d %d %d\n", n,m,oriented);
//#endif

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


void do_asp(int **tab, int **next, int n)
{
	int i, j, k, tmp;

	for (k = 0; k < n; k++) {
		for (i = 0; i < n; i++) {
			if (i != k) {
				for (j = 0; j < n; j++) {
					tmp = tab[i][k] + tab[k][j];
					if (tmp < tab[i][j]) {
						tab[i][j] = tmp;
            next[i][j] = next[i][k];
					}
				}
			}
		}
	}
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

int main(int argc, char **argv)
{
	int n,m, bad_edges=0, oriented=0, lb, ub, i, city_1, city_2;
	int **tab, **next;
	int print = 0;
  int get = 0;
  int diameter = 0;
  int random = 0;
	struct timeval start;
	struct timeval end;
	double time;
	char FILENAME[100];


	usage();

	n = 0;
  if(argc < 3){
    exit(1);
  }

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

      i+=2;
      continue;
    }
    if (!strcmp(argv[i], "-diameter")){
      diameter = 1;
      continue;
    }
	}
	if (random) {
		init_tab(n,&m,&tab,oriented); // last one = oriented or not ... 
	}
	else bad_edges = read_tab(FILENAME, &n, &m, &tab, &oriented); 

	
	fprintf(stderr, "Running ASP with %d rows and %d edges (%d are bad)\n", n, m, bad_edges);

	lb = 0;    /* lower bound for rows to be computed */
	ub = n;    /* upper bound for rows to be computed */

//	init_tab(n, &tab);


  init_next(n, &next);

	if(gettimeofday(&start, 0) != 0) {
		fprintf(stderr, "could not do timing\n");
		exit(1);
	}

	do_asp(tab, next, n);
  if(get){
    get_path(city_1, city_2, next, tab, n);
  }


	if(gettimeofday(&end, 0) != 0) {
		fprintf(stderr, "could not do timing\n");
		exit(1);
	}

	time = (end.tv_sec + (end.tv_usec / 1000000.0)) - 
		(start.tv_sec + (start.tv_usec / 1000000.0));

	fprintf(stderr, "ASP took %10.3f seconds\n", time);

	if(print == 1) {
		print_tab(tab, n);
	}


  if(diameter){
    diameter = get_diameter(tab, n);
    printf("The diameter of the network is: %d\n", diameter);
  }

	free_tab(tab, n);
  free_tab(next ,n);

	return 0;
}
