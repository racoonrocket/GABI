#include "bigWig.h"
int main(int argc, char *argv[]) {
    bigWigFile_t *fp = NULL;
    bwOverlappingIntervals_t *intervals = NULL;
    double *stats = NULL;
    if(argc != 2) {
        fprintf(stderr, "Usage: %s {file.bw|URL://path/file.bw}\n", argv[0]);
        return 1;
    }

    //Initialize enough space to hold 128KiB (1<<17) of data at a time
    if(bwInit(1<<17) != 0) {
        fprintf(stderr, "Received an error in bwInit\n");
        return 1;
    }

    //Open the local/remote file
    fp = bwOpen(argv[1], NULL, "r");
    if(!fp) {
        fprintf(stderr, "An error occured while opening %s\n", argv[1]);
        return 1;
    }

    //Get values in a range (0-based, half open) without NAs
    intervals = bwGetValues(fp, "chr1", 10000000, 10000100, 0);
    bwDestroyOverlappingIntervals(intervals); //Free allocated memory
    }

    bwClose(fp);
    bwCleanup();
    return 0;
}