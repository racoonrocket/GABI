
import GABI
import time
import numpy as np

import GABI as gb
chrs = ['chr'+str(i) for i in range(1,22)]
gbt = gbimp.GABI(yamfile="/zslow0/marnier/stage/HK_stud/H3K27ac/result.yaml",bw=True,chr_list=chrs)
gbt.fit()
matrixCT,statemat,labelstate = gbt.predict()
#print("matrixCT",matrixCT)
gbt.save_as_bigwig(matrixCT)



