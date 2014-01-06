#include "init.h"

void parseUserInput(int argc, char** argv) {
	int c;
	char configfile[30];
	int outfileSpecified = 0;
	int configfileSpecified = 0;
	int typeSpecified = 0;
	if (argc == 1) {
		fprintf(stderr, "GTgraph-random [-options]\n");
		fprintf(stderr, "\t-t ### graph model\n");
		fprintf(stderr, "\t        (0 for star, 1 for chain\n"); 
		fprintf(stderr, "\t-s ### size=radius in hops\n");
		fprintf(stderr, "\t-d ### degree=no. of neighbors per star (graph model 0)\n");	
		fprintf(stderr, "\t-o ###  output file to write the graph to\n");
		fprintf(stderr, "\t        (default: sample.gr)\n"); 
		fprintf(stderr, "\t-h      display this message\n\n");
		fprintf(stderr, "No config file specified\n");
		fprintf(stderr, "Assigning default values from init.c\n");
		getParams();
		updateLog();

	} else if (argc <= 9) {
		
		while ((c = getopt(argc, argv, "c:t:s:d:o:h:")) != -1) {

			switch (c) {
				
				case 't':
				if (configfileSpecified) 
					break;
				if (!outfileSpecified) {
					getParams();
				}
				GRAPH_MODEL = atoi(optarg);
				typeSpecified = 1;
                                fprintf(stderr, "Graph model %d chosen\n", GRAPH_MODEL);
                                updateLog();
				break;
					
				case 'o':
				outfileSpecified = 1;
				if ((!configfileSpecified) && (!typeSpecified)) {
					fprintf(stderr, "No config file specified, assigning default values ...\n");
					getParams();
				}
				WRITE_TO_FILE = 1;
				strcpy(OUTFILE, optarg);
				fprintf(stderr, "Graph will be written to %s\n", OUTFILE);
				strcpy(NEWFILE, optarg);
				strcat(NEWFILE, ".new");
				updateLog();
				break;
				
				case 'c':
				fprintf(stderr, "Warning: The parameters specified in the config file will be applied and all other options will be lost\n");
				configfileSpecified = 1;
				if (!outfileSpecified) {
					getParams();
					strcpy(configfile, optarg);
					fprintf(stderr, "Reading config file %s\n", configfile);
					getParamsFromFile(configfile);
				} else {
					strcpy(configfile, optarg);
					fprintf(stderr, "Updating parameters from config file %s\n", configfile);
					getParamsFromFile(configfile);
				}
				updateLog();
				break;
		
				case 's':
				if (configfileSpecified)
					break;
				if (!typeSpecified) {
					fprintf(stderr, "Error! First specify graph model using the -t option before this argument\n");
					exit(-1);
				}

				size = atoi(optarg);
				fprintf(stderr, "size is set to %d\n", size);
				updateLog();
				break;

				case 'd':
				if (configfileSpecified)
                                        break;
                                if (!typeSpecified) {
                                        fprintf(stderr, "Error! First specify graph model using the -t option before this argument\n");
                                        exit(-1);
                                }

                                out_degree = atoi(optarg);
                                fprintf(stderr, "out_degree is set to %d\n", out_degree);
                                updateLog();
				break;

				case 'h':
				usage();

				default:
				usage();
			}
		}

		
	} else {
		fprintf(stderr, "Invalid input arguments\n");
		usage();
	}
	
}

void usage() {
                fprintf(stderr, "GTgraph-random [-options]\n");
                fprintf(stderr, "\t-t ### graph model\n");
                fprintf(stderr, "\t        (0 for star, 1 for chain\n");
                fprintf(stderr, "\t-size ###  radius in hops\n");
                fprintf(stderr, "\t-degree ###  no. of neighbors per star (graph model 0)\n");
                fprintf(stderr, "\t-o ###  output file to write the graph to\n");
                fprintf(stderr, "\t        (default: sample.gr)\n");
                fprintf(stderr, "\t-h      display this message\n\n");
                fprintf(stderr, "No config file specified\n");
                fprintf(stderr, "Assigning default values from init.c\n");

	exit(-1);
}

/* Default Input parameters for graph generation. These values can 
 * also be specified in a configuration file and passed as input to 
 * the graph generator */
void getParams() {

	GRAPH_MODEL = 0;
	ORIENTED = 0;
	size = 1;
	out_degree = 10;
	MAX_WEIGHT = 100; 
	MIN_WEIGHT = 0;

	STORE_IN_MEMORY = 1;
	
	SORT_EDGELISTS = 1;
	SORT_TYPE = 0;
	WRITE_TO_FILE = 1;

	strcpy(OUTFILE, "sample.gr");	
        strcpy(NEWFILE, "sample.gr.new");
	strcpy(LOGFILE, "log");	
}

void getParamsFromFile(char* configfile) {

	/* read parameters from config file */
	FILE *fp;
	char line[128], var[32];
	double val;

	fp = fopen(configfile,"r");
	if (fp == NULL) {
		fprintf(stderr, "Unable to open config file:%s\n",configfile);
		exit(-1);
	}

	while (fgets(line, sizeof (line), fp) != NULL) {
		sscanf(line,"%s %lf",var, &val);
		if (*var == '#') continue;  /* comment */
		else if (strcmp(var,"GRAPH_MODEL")==0) {
			GRAPH_MODEL = (int) val;
		} else if (strcmp(var,"ORIENTED")==0) {
                        GRAPH_MODEL = (int) val;
		} else if (strcmp(var,"size")==0) {
			size = (int) val;
		} else if (strcmp(var,"degree")==0) {
			out_degree = (int) val;
		} else if (strcmp(var,"MAX_WEIGHT")==0) {
			MAX_WEIGHT = (WEIGHT_T) val;
		} else if (strcmp(var,"MIN_WEIGHT")==0) {
			MIN_WEIGHT = (WEIGHT_T) val;
		} else if (strcmp(var,"STORE_IN_MEMORY")==0) {
			STORE_IN_MEMORY = (int) val;
		} else if (strcmp(var,"SORT_EDGELISTS")==0) {
			SORT_EDGELISTS = (int) val;
		} else if (strcmp(var,"SORT_TYPE")==0) {
			SORT_TYPE = (int) val;
		} else if (strcmp(var,"WRITE_TO_FILE")==0) {
			WRITE_TO_FILE = (int) val;
		} else {
			fprintf(stderr,"unknown parameter: %s\n",line);
		}
	}

}
