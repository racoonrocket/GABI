# GABI
Genome Annotation using Bayesian Inference (GABI)

## How to use:

### Simple Core Version
inputs:
 - matrix: MxN binary matrix
 - labels: M dim vector of profiles clusters (or cell type) membership
 
output:
 - matrixCT: (CxN binary matrix with C clusters (or cell types) ) Cleaned matrix
 - matrixCTProba: (CxN binary matrix with C clusters (or cell types) ) Cleaned matrix probability to have a peak
 - combmat: (CxK binary matrix with K combinations) Combinations between the cell types
 - statemat: (MxK binary matrix with K combinations) Combinations between the cell types with all samples
 - labels_states: (N integer vector) combinations position along the genome
 
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

Additional inputs:
 - distmat: (MxM distance matrix) Distance between the profiles. If empty, yule distance is used.
 - Nclust: Number of cell types per slices
 
```
import GABI_MP as gbimp

gb = gbimp.GABI(matrix,labels,distmat=[],verbose=True,NClust=8)
gb.fit()
matrixCT,statemat,labels_states = gb.predict()

#Additional:
combmat = gb.combmat

```

## Note
KeyboardInterrupt is handled in both version if we want to end the fit before the end.
