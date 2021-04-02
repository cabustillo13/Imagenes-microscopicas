# -*- coding: utf-8 -*-

""" Determinar cómo varía el área plana de la imagen a lo largo del tiempo"""

import matplotlib.pyplot as plt
from skimage import io, restoration
from skimage.filters.rank import entropy
from skimage.filters import try_all_threshold, threshold_otsu
from skimage.morphology import disk
import numpy as np

img = io.imread("./Imagenes/entropia/Scratch0.jpg")

##############
## Entropía ##
##############
# La entropía se utiliza mucho en la detección de texturas
# Al hacer más grande el argumento de disk(), se va difuminando más la imagen
entropy_img = entropy(img, disk(3))
plt.imshow(entropy_img, cmap= "gray")
plt.show()

####################################################
## Determinar cuál threshold me conviene utilizar ##
####################################################
fig, ax = try_all_threshold(entropy_img, figsize=(10,8), verbose= False)
plt.show()
# Se concluye que se puede usar cualquiera: Minimum, Otsu, Mean o Isodata

th = threshold_otsu(entropy_img)    # Devuelve un flotante a th
binary = entropy_img <= th          # Convertir la imagen a binario
plt.imshow(binary, cmap= "gray")
plt.show()

#############################################################################
## Determinar la cantidad de píxeles blancos -> Permite determinar el área ##
#############################################################################
# Como quiero porcentaje se hace (no._pixelesBlancos / (no._pixelesBlancos + no._pixelesNegros))*100
print("El porcentaje de píxeles blancos es: ", (np.sum(binary==1)/(np.sum(binary==1) + np.sum(binary==0)))*100)
