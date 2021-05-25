from BW2GABI import bigwig2gabi
import GABI_MP as gbimp
import time
import numpy as np

import GABI as gb
gbt = gb.GABI(bw=True,chr_list=['chr3'])

gbt.fit()
matrixCT = gbt.predict()
#print("matrixCT",matrixCT)
gbt.save_as_bigwig(matrixCT)



