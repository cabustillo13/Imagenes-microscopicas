# -*- coding: utf-8 -*-

"""Análisis de la variación en celulas mediante análisis de texturas"""
""" Paper implementado: https://pubmed.ncbi.nlm.nih.gov/25482647/"""

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

# Me va a servir para leer varias imágenes dentro de una carpeta
import glob 

# Realizar una gráfica %área vrs tiempo 
time = 0
time_list = []
porcentaje_list = []
path = "./Imagenes/entropia/*.*"

for file in glob.glob(path):
    
    img = io.imread(file)
    entropy_img = entropy(img, disk(3))
    th = threshold_otsu(entropy_img)
    binary = entropy_img <= th
    porcentajeArea = (np.sum(binary==1)/(np.sum(binary==1) + np.sum(binary==0)))*100
    print("El porcentaje de píxeles blancos es: ", porcentajeArea)
    
    time_list.append(time)
    porcentaje_list.append(porcentajeArea)
    
    time+=1
    
plt.plot(time_list, porcentaje_list, "bo")
plt.show()

# Para realizar una regresión lineal
from scipy.stats import linregress

slope, intercept, r_value, p_value, stderr = linregress(time_list, porcentaje_list)
#print(linregress(time_list, porcentaje_list)) # Devuelve la función

# Regresion Lineal
print("y = ", slope, "x", " + ", intercept)

# Mínimos cuadrados
print("R\N{SUPERSCRIPT TWO} = ", r_value**2)

