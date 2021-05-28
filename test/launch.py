
import GABI
import time
import numpy as np

chrs = ['chr3']
gbt = GABI.multicore(yamfile='sources.yaml',bw=True,chr_list=chrs)
gbt.fit()
matrixCT,statemat,labelstate = gbt.predict()
#print("matrixCT",matrixCT)
gbt.save_as_bigwig(matrixCT,prename="")



