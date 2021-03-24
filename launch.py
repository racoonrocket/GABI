import GABI as gbi
import pandas as pd
import numpy as np
import random as rd


unic = [rd.randint(0,1) for k in range(1000)]
print(unic)
b = [0]*2
b = pd.DataFrame(b)

matrice = []
for i in range(2):
    variant = []
    for k in range(len(unic)):
        if unic[k] == 0:
            variant.append(0)
        else:
            variant.append(1)
    for k in range(500):
        indice = rd.randint(0,999)
        if variant[indice] == 0:
            variant[indice] = 1
        else:
            variant[indice] = 0
    matrice.append(variant)

a = pd.DataFrame(matrice)
print(a)


gb = gbi.GABI(b)
gb.fit(a)
matrixCT = gb.predict(a)
print("ENNDEED")
print(matrixCT)

count = 0
for i in range(len(matrixCT[0])):
    if matrixCT[0][i] != unic[i]:
        print(str(matrixCT[0][i]) + " + " + str(unic[i]))
        count+=1
print(count)
print(len(matrixCT[0]))
print(len(unic))