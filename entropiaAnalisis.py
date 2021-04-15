# -*- coding: utf-8 -*-

"""Análisis de la variación en celulas mediante análisis de texturas"""
""" Paper implementado: https://pubmed.ncbi.nlm.nih.gov/25482647/"""

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

import cv2
import os

# Me va a servir para leer varias imágenes dentro de una carpeta
import glob 

# Escribir dentro de la imagen los resultados
def visualizeImg(img, size, time):
    # Redondear el porcentaje % a 2 cifras significativas
    cv2.putText(img, "Size: "  + str(round(size,2)) + "%", (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, str(time), (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.imwrite("./Resultados/imagen"+str(time)+".jpg", img)
    
# Crear un pequeño vídeo uniendo varias imágenes
def makeVideo():
    
    image_folder = "./Resultados/"
    video_name = 'demo.avi'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    # Hay que ordenar las imágenes de manera ascendente para tener los resultados correctos esperados
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 2, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Realizar una gráfica %área vrs tiempo 
time = 0
time_list = []
porcentaje_list = []
path = "./Imagenes/entropia/*.*"

# Ordenar de manera ascendente los archivos
archivos = glob.glob(path)
archivos.sort()

for file in archivos:
    
    img = io.imread(file)
    entropy_img = entropy(img, disk(3))
    th = threshold_otsu(entropy_img)
    binary = entropy_img <= th
    porcentajeArea = (np.sum(binary==1)/(np.sum(binary==1) + np.sum(binary==0)))*100
    print("El porcentaje de píxeles blancos es: ", porcentajeArea)
    
    time_list.append(time)
    porcentaje_list.append(porcentajeArea)
    
    # Escribir en la imagen el número de imagen y el % de píxeles
    visualizeImg(img, porcentajeArea, time)
    
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

# Crear video de demostración
makeVideo()
