# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:38:44 2020

@author: Paula
"""
import cv2
import sys

import numpy as np

from matplotlib import pyplot as plt
#from skimage.feature import hog


import SeamCarving
import energias



def representar_imagenes(lista_imagen_leida, lista_titulos, n_col=2, tam=15):

    # Comprobamos que el numero de imágenes corresponde con el número de títulos pasados
	if len(lista_imagen_leida) != len(lista_titulos):
		print("No hay el mismo número de imágenes que de títulos.")
		return -1 # No hay el mismo numero de imágenes que de títulos

	# Calculamos el numero de imagenes
	n_imagenes = len(lista_imagen_leida)

	# Calculamos el número de filas
	n_filas = (n_imagenes // n_col) + (n_imagenes % n_col)

	# Establecemos por defecto un tamaño a las imágenes
	plt.figure(figsize=(tam,tam))

	# Recorremos la lista de imágenes
	for i in range(0, n_imagenes):

		plt.subplot(n_filas, n_col, i+1) # plt.subplot empieza en 1

		if (len(np.shape(lista_imagen_leida[i]))) == 2: # Si la imagen es en gris
			plt.imshow(lista_imagen_leida[i], cmap = 'gray')
		else: # Si la imagen es en color
			plt.imshow(cv2.cvtColor(lista_imagen_leida[i], cv2.COLOR_BGR2RGB))

		plt.title(lista_titulos[i]) # Añadimos el título a cada imagen

		plt.xticks([]), plt.yticks([]) # Para ocultar los valores de tick en los ejes X e Y

	plt.show()


def verticalSeam (image, energy):

    n, m = image.shape[:2]


    M = energy.copy()  # Matriz para la energía acumulativa mínima

    # Recorremos la imagen desde la segunda fila hasta la última
    for i in range (1, n):

        if m > 1:

            M[i,0] = energy[i,0] + min(M[i-1, 0], M[i-1,1])

        else:
            M[i,0] = energy[i,0] + M[i-1, 0]

        for j in range (1, m-1):

            M[i,j] = energy[i,j] + min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])

        M[i,m-1] = energy[i,m-1] + min(M[i-1, m-2], M[i-1,m-1])

    return SeamCarving.crearCamino (M)

img = SeamCarving.readImage("harry.jpg", 1)
positivo = SeamCarving.readImage("positivo_harry.jpg", 0)
negativo = SeamCarving.readImage("negativo_harry.jpg", 0)

n, m = img.shape[:2]
scale_factor = 0.5
height = int(n * scale_factor)
width = int (m * scale_factor)
dim = (width, height)

img = cv2.resize(img, dim)
negativo = cv2.resize(negativo, dim)
positivo = cv2.resize(positivo, dim)

positivo = positivo.astype(np.float)
negativo = negativo.astype(np.float)

for i in range (250):

    energia = energias.forwardEnergy(img)

    energia = SeamCarving.removeEnergy(energia, negativo)

    energia = SeamCarving.preserveEnergy(energia, positivo)

    a, b, camino = SeamCarving.Seam(img, energia)

#    image = SeamCarving.drawSeams ([camino], [], image)
    img = SeamCarving.removeSeam (img, camino)
    negativo = SeamCarving.removeSeam (negativo, camino)
    positivo = SeamCarving.removeSeam (positivo, camino)


#image = image.astype(np.uint8)
representar_imagenes([img], ["edit"])
