# GABI
Genome Annotation using Bayesian Inference (GABI) is a tool
that allow to treat epigenomic data after peak calling
in the case you have not summed your repetitions.


![Alt text](Capture_GABI.png)

##Requirements :
 - numpy 1.20.1
 - scipy 1.6.1
 - pyBigWig 0.3.18
 - tqdm 4.59.0
 

## How to use:

### Simple Core Version
#### Inputs:
 - **matrix**: MxN binary matrix (boolean values only), can be sparsed (CSR) or dense. 
 - **labels**: M dim vector of profiles clusters (or cell type) membership 
 /!\ The labels vector must start from zero and only contains integer
 - **example** : you have 4 differents clusters of respectively 2,2,1,1 epigenomic profiles,each profile contains 30 values. your
  labels vector (6*1) & and your matrix (30*6)  will look like : 
  ``` 
matrix = numpy.array([True,False,...,False],[True,False,...,True],[True,True,...,True],[True,True,...,False],[,False,...,False],[True,False,...,False])
labels = numpy.array([0,0,1,1,2,3])
```
#### Output:
 - **matrixCT**: (CxN binary matrix with C clusters (or cell types) ) Cleaned matrix
 - **matrixCTProba**: (CxN binary matrix with C clusters (or cell types) ) Cleaned matrix probability to have a peak
 - **combmat**: (CxK binary matrix with K combinations) Combinations between the cell types
 - **statemat**: (MxK binary matrix with K combinations) Combinations between the cell types with all samples
 - **labels_states**: (N integer vector) combinations position along the genome
 
 #### How to Run GABI:
 
```
import GABI as gbi

gb = gbi.GABI(labels)
gb.fit(matrix)
matrixCT = gb.predict(matrix)

#Additional
labels_states = gb.labels_state
statemat = gb.statemat
matrixCTProba = gb.predict(matrixC,GetProba=True)

```

### Multiprocessing Version
The multiprocessing version slice the samples in groups of similar cell types and apply GABI on each part before merging them.

#### Additional inputs:
 - **distmat**: (MxM distance matrix) Distance between the profiles. If empty, yule distance is used.
 - **Nclust**: Number of cell types per slices
 
```
import GABI_MP as gbimp

gb = gbimp.GABI(matrix,labels,distmat=[],verbose=True,NClust=8)
gb.fit()
matrixCT,statemat,labels_states = gb.predict()

#Additional:
combmat = gb.combmat

```
### Using BigWig as input
If your profiles are in another format you can convert them to BigWigs and directly use
GABI with it. <br>
/!\ In that case, a peak calling must have been already applied to your data
and your bigwigs must be binary ( eg : contain 0 and 1 or True and False)
![Alt text](drawio.jpg)
#### Steps :
 1) Fill the sources.yaml file with path to the BigWigs and their cluster appartenance
 2) When you call GABI, dont add any matrix, set the bw param to True and specify chromosomes 
 that you want to put in your matrix
 
#### 1. YAML file
The first step to use GABI with big wig is to fill the sources.yaml file.
<br>
If you have 2 different type of cells, with 3 repetition each, you have 6 bigwigs
In that case you must add the 6 paths to the sources.yaml file in that way:
```
0:
  - /clusterone/path/to/the/first.bw
  - /clusterone/path/to/the/second.bw
  - /clusterone/path/to/the/third.bw
1:
  - /clustertwo/path/to/the/first.bw
  - /clustertwo/path/to/the/second.bw
  - /clustertwo/path/to/the/third.bw
``` 
Now imagine that you have a third cell type with two repetition in it
the sources.yaml become:
```
0:
  - /clusterone/path/to/the/first.bw
  - /clusterone/path/to/the/second.bw
  - /clusterone/path/to/the/third.bw
1:
  - /clustertwo/path/to/the/first.bw
  - /clustertwo/path/to/the/second.bw
  - /clustertwo/path/to/the/third.bw
2:
  - /clusterthree/path/to/the/first.bw
  - /clusterthree/path/to/the/second.bw

```
ect...
#### 2. Run GABI with bigwigs as input
If you want to only have the chromosme name chr3 in your BigWig to
end up in your genomic profile you need to call GABi like that:
```
import GABI as gb
gbt = gb.GABI(bw=True,chr_list=['chr3'])

gbt.fit()
matrixCT = gbt.predict()
#print("matrixCT",matrixCT)
gbt.save_as_bigwig(matrixCT)
```
or with Multiprocessing version
```
import GABI_MP as gbimp
gbt = gbimp.GABI(bw=True,chr_list=['chr3'])

gbt.fit()
matrixCT,statemat,labels_states = gbt.predict()
#print("matrixCT",matrixCT)
gbt.save_as_bigwig(matrixCT)
```
If you dont know what chromosomes are in your bigwig you can use
pyBigWig : 
```
import pyBigWig
bw = pyBigWig.open("path/to.bw")
print(bw.chroms())
```
## Note
KeyboardInterrupt is handled in both version if we want to end the fit before the end.
