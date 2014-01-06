#include "graph.h"

void graphGen(graph* g)
{

	LONG_T *startVertex, *endVertex;
	WEIGHT_T *weights;
	LONG_T i, j, k, crt, level;
	WEIGHT_T w;
	LONG_T estNumEdges, numEdges, edgeNum;	
	int *stream1, *stream2;
//	FILE *outfp, *newfp;
	int oriented = ORIENTED;

	/*----------------------------------------------*/
	/*		initialize SPRNG 		*/
	/*----------------------------------------------*/

	stream1 = init_sprng(SPRNG_CMRG, 0, 1, SPRNG_SEED1, SPRNG_DEFAULT);
	stream2 = init_sprng(SPRNG_CMRG, 0, 1, SPRNG_SEED2, SPRNG_DEFAULT);

	/*------------------------------------------------------*/
	/*		generate edges as per the		*/		 
	/*		graph model and user options	      	*/
	/*------------------------------------------------------*/

	if (GRAPH_MODEL == 0) {
// 		out_degree = number of neighbours per star 
//		size = number of stars 
		printf("%d %d \n", size, out_degree);
		n=1; m=1;  
		for (i=0; i<size; i++) {
			 m*=out_degree; 
			 n+= m;
		}
		m=n-1;
		numEdges = m; 
		startVertex = (LONG_T *) malloc(m * sizeof(LONG_T));
		endVertex   = (LONG_T *) malloc(m * sizeof(LONG_T));
		weights	    = (LONG_T *) malloc(m * sizeof(LONG_T));
		crt = 0; i=0; k=0;
		while (crt<m && i<n) {
			for (j=0; j<out_degree; j++) {
				w = (LONG_T) isprng(stream1) % n;
				startVertex[crt]=i; 
				endVertex[crt]=k+j+1; 
				weights[crt]=w;
				crt++;
			}
			k+=out_degree;
			i++;
		}		
	}
	else {
//                printf("%d %d \n", size, out_degree);
		out_degree = 1; //chain !! 
                n=size; m=n-1;
                numEdges = m;
                startVertex = (LONG_T *) malloc(m * sizeof(LONG_T));
                endVertex   = (LONG_T *) malloc(m * sizeof(LONG_T));
                weights     = (LONG_T *) malloc(m * sizeof(LONG_T));
		for (i=0; i<m; i++) {
			w = (LONG_T) isprng(stream1) % n;
			startVertex[i]=i;
			endVertex[i]=i+1;		
			weights[i]=w;
		}
	}
 	


	fprintf(stderr, "done\n");

	free(stream1);

        /*-------------------------------------------------------*/
        /*              sort the edge lists with start           */
        /*              vertex as primary key                    */
        /*---------------------------------------------- --------*/

        if (SORT_EDGELISTS) {

                fprintf(stderr, "Sorting edge list by start vertex ... ");
		if (GRAPH_MODEL == 1) {
                	if (SORT_TYPE == 0) {
                        	/* Counting sort */
                        	countingSort(startVertex, endVertex, weights, numEdges);
                	} else {
                        	/* Heap sort */
                       		heapSort(startVertex, endVertex, weights, numEdges);
                	}
		}
                fprintf(stderr, "done\n");
        }

        g->start = startVertex;
        g->end = endVertex;
        g->w = weights;
        g->n = n;
        g->m = numEdges;
}
