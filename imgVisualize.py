# -*- coding: utf-8 -*-

"""Análisis de la variación en celulas mediante análisis de texturas"""
""" Paper implementado: https://pubmed.ncbi.nlm.nih.gov/25482647/"""

import cv2
# Me va a servir para leer varias imágenes dentro de una carpeta
import glob 

# Realizar una gráfica %área vrs tiempo 
time = 0
time_list = []
porcentaje_list = []
path = "./Imagenes/entropia/*.*"

# Ordenar de manera ascendente los archivos
archivos = glob.glob(path)
archivos.sort()
print(archivos)

for file in archivos:
    
    img = cv2.imread(file)
    
    # Escribir en la imagen el número de imagen y el % de píxeles
    cv2.putText(img, "Size: "  + str(time), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, str(time), (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.imwrite("./Resultados/imagen"+str(time)+".jpg", img)
    time+=1
    

# Realizar un video concatenando varias imagenes
import os

image_folder = "./Resultados/"
video_name = 'video.avi'

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
