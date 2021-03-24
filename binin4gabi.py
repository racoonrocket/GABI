import pyBigWig as bg

adrr = "bigWigExample.bw"
bw = bg.open(adrr) #open the big wig
chr = bw.chroms()
print(chr)
binned = bw.stats("chr21",1,chr["chr21"],nBins=50)
print(binned)