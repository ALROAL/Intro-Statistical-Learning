archivo = input("Introduce el nombre del archivo: " )
values = input("Introduce el nombre de la columna de datos: ")

import numpy as np
import pandas as pd
from scipy import stats


df = pd.read_excel(archivo,"Sheet1")

cols = list(df)
cols.remove(values)

factores = {}

for i in cols:
    factores[i] = pd.unique(df[i])
    #print(factores[i])

tabla = {}

for i in cols:
    for j in factores[i]:
        tabla[j] = df[df[i].isin([j])][[i,values]].reset_index(drop = True)

h = -1
k = 0
        
for i in range(len(cols)):
    
    h = h + 1    
    tabla[i+i] = df[(df[cols[0]].isin([factores[cols[0]][h]]))]
        
    for j in cols:
            
        if j!=i:
                
            for k in range(len(factores[j])):
                    
                tabla[factores[cols[0]][h]+factores[j][k]] = tabla[i+i][tabla[i+i][j].isin([factores[j][k]])].reset_index(drop = True)

ss = {}

for j in cols:
    ss[j] = 0

for j in cols:
    for i in range(len(factores[j])):
        for k in range(len(tabla[factores[j][i]])):
            ss[j] = ss[j] + (tabla[factores[j][i]][values].mean()-df[values].mean())**2
                
#for j in cols:
    #print(j)
    #print(ss[j])

sserror = 0

for i in (cols[0],):
    for j in range(len(factores[i])):
        for k in cols[::-1]:
            if i!= k:
                for h in range(len(factores[k])):
                    for n in range(len(tabla[factores[i][j]+factores[k][h]])):
                        sserror = sserror + (tabla[factores[i][j]+factores[k][h]][values][n] - tabla[factores[i][j]+factores[k][h]][values].mean())**2
#print(sserror)

sst = 0

for i in range(len(df)):
    sst = sst + (df[values][i] - df[values].mean())**2
#print((sst))

ssfactores = sst - sserror

for j in cols:
    ssfactores = ssfactores - ss[j]
#print(ssfactores)



dfr = {}

for j in cols:
    dfr[j] = 0

for j in cols:
    dfr[j] = len(factores[j]) - 1


dfrfactores = 1
for j in cols:
    dfrfactores = dfrfactores*dfr[j]

dferror = 0
for i in (cols[0],):
    for j in range(len(factores[i])):
        for k in cols[::-1]:
            if i!= k:
                for h in range(len(factores[k])):
                    dferror = dferror + (len(tabla[factores[i][j]+factores[k][h]])-1)

dft = dferror + dfrfactores
for j in cols:
    dft = dft +dfr[j]

#print(dfr["Gender"],dfr["Age Group"], dfrfactores, dferror, dft)

denom = sserror/dferror

fval = {}

for j in cols:
    fval[j] = 0
for j in cols:
    fval[j] = (ss[j]/dfr[j])/denom

fvalfactores = (ssfactores/dfrfactores)/denom

pval = {}

for j in cols:
    pval[j] = 0
    
for j in cols:
    pval[j] = stats.f(dfr[j],dferror).sf(fval[j])

pvalfactores = stats.f(dfrfactores,dferror).sf(fvalfactores)


for j in cols:
	print(("El p-valor del factor {} es: {}".format(j,pval[j])))

print(("El p-valor de la interacción de los factores es: {}".format(pvalfactores)))