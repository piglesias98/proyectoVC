# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:38:44 2020

@author: Paula
"""
import cv2

import numpy as np

from matplotlib import pyplot as plt
#from skimage.feature import hog


import SeamCarving




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







# Prueba de funcionamiento


#simple_energy = SeamCarving.simpleEnergy(img)
#simple_energy2 = SeamCarving.eHOG(img)

#img2 = cv2.cvtColor(simple_energy, cv2.COLOR_BGR2RGB)
#simple_energy2 = cv2.cvtColor(simple_energy2, cv2.COLOR_BGR2RGB)


#for i in range(500):
#        for j in range(500):
#            if i-100<0: inicio_i=0
#            else: inicio_i = i-100
#            if i+101>image.shape[0]: fin_i = image.shape[0]
#            else: fin_i = i+101
#
#            if j-100<0: inicio_j=0
#            else: inicio_j = j-5
#            if j+101>image.shape[0]: fin_j = image.shape[1]
#            else: fin_j = j+101
#
#            window = image[inicio_i:fin_i,inicio_j:fin_j]
#            imagenes.append(window)
#            titulos.append("windo2")




#window = img[0:100, 0:100]
#hog = cv2.HOGDescriptor()
#h = hog.compute(img)
#print(h)
#hogg, img2 = hog(img, orientations = 8, visualize=True)
#print(hogg)
#representar_imagenes([img2],["titulos"])


#matriz = np.matrix('1, 2, 3, 4; 3, 5, 6, 7; 8, 9, 10, 11; 12, 13, 14, 15' )
#print(matriz)
#print(matriz.shape[0])
#maximos=np.zeros(matriz.shape)
#for i in range(matriz.shape[0]):
#    for j in range(matriz.shape[1]):
#        print("elemento", matriz[i,j])
#        if i-1<0: inicio_i=0
#        else: inicio_i = i-1
#        if i+2>matriz.shape[0]: fin_i = matriz.shape[0]
#        else: fin_i = i+2
#
#        if j-1<0: inicio_j=0
#        else: inicio_j = j-1
#        if j+2>matriz.shape[0]: fin_j = matriz.shape[1]
#        else: fin_j = j+2
#
#        a = matriz[inicio_i:fin_i,inicio_j:fin_j]
#        maximos[i,j] = np.max(a)
#        print("ventana", a)
#print("maximos",maximos)

#cv2.imshow("original", image)
#cv2.imshow("energia 1", simple_energy)
#cv2.imshow("energia 2", simple_energy2)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
<<<<<<< HEAD
#img = SeamCarving.readImage("playa.jpg", 1)
=======

#img = SeamCarving.readImage("arco.jpg", 1)
#
#img_v = img.copy()
#img_f = img.copy() 
#
#image_v = img.copy()
#image_f = img.copy()
#
#for i in range (100):
#    
#    camino_v = SeamCarving.verticalSeam(image_v, SeamCarving.simpleEnergyRGB)
#    camino_f = SeamCarving.verticalSeam(image_f, SeamCarving.forwardEnergy)
#    img_v = SeamCarving.drawSeams ([camino_v], [], img_v)
#    img_f = SeamCarving.drawSeams ([camino_f], [], img_f)
#    
#    image_v = SeamCarving.removeSeam (image_v, camino_v)
#    image_f = SeamCarving.removeSeam (image_f, camino_f)
#
##image = SeamCarving.eHOG (img)
##simple = SeamCarving.simpleEnergy(img)
##RGB = SeamCarving.simpleEnergyRGB(img)
##
##hog1 = SeamCarving.eHOG(img, 1)
##hog2 = SeamCarving.eHOG(img, 0)
##
##representar_imagenes([simple, RGB], ["simple", "RGB"])
#representar_imagenes([img_v, img_f], ["original", "image"])
#representar_imagenes([image_v, image_f], ["original", "image"])



>>>>>>> Laura-DEV
#nn = 200
#nm = 400
#
#imagenes=[]
#n, m = img.shape[:2]
#print("n", n)
#print("m", m)
#print("nn",nn)
#print("nm", nm)
#print("nn/n", nn/n)
#print("nm/m", nm/m)
#scale_factor = max(nn/n, nm/m)
#print(scale_factor)
#height = int(n * scale_factor)
#print("hegiht", height)
#width = int (m * scale_factor)
#print("widht", width)
#dim = (width, height)
## resize image
#resized = cv2.resize(img, dim)
#print("resized schape", resized.shape)
##Eliminamos las verticales o horizontales que sobren
#print("sobran ", height - nn, " horizontales")
#print("sobran ", width - nm, " verticales")
#for i in range(height - nn):
#    rotada = np.rot90(resized, 1)
#    a, b, path = SeamCarving.verticalSeam(rotada)
#    rotada = SeamCarving.removeSeam (rotada, path, 1)
#    resized = np.rot90(rotada, 1)
#for i in range(width - nm):
#    a, b, path = SeamCarving.verticalSeam(resized)
#    resized = SeamCarving.removeSeam(resized, path, 1)
#print("shape final",resized.shape)
#imagenes.append(resized)
#
#img4 = SeamCarving.readImage("playa.jpg", 1)
#
#
#T, options = SeamCarving.seamsOrder(img4, 200, 400)
#order = SeamCarving.selectSeamsOrder (img4, T, options)
#seams = SeamCarving.removeOrderSeams (img4, order)
#
#imagenes.append(seams)
#representar_imagenes(imagenes,["resizes", "seams"],1)
#scale_factor = max(nn/n, nm/m)
#print("nn/n", nn/n, "nm/m", nm/m)
#print("factor", scale_factor)
#height = int(n * scale_factor)
#print("height", height)
#width = int (m * scale_factor)
#print("width", width)
#dim = (width, height)
## resize image
#resized = cv2.resize(image, dim)
#print("resized shape", resized.shape)
#representar_imagenes([image, resized], ["image", "resized"])

#T, options = SeamCarving.seamsOrder(img, 200, 400)
#order = SeamCarving.selectSeamsOrder (image, T, options)

#img1 = SeamCarving.addOrderSeams (image, order)
#img2 = SeamCarving.scaleAndRemoveSeams(image, 200, 400)
#img1 = SeamCarving.removeOrderSeams (img, order)
#

#
#representar_imagenes([img1, img2],["sin scale", "con scale"],1)
<<<<<<< HEAD
#    
#from skimage.feature import hog
#
#
#img = SeamCarving.readImage("playa.jpg", 1)
#print(img.shape)
#
#e1 = SeamCarving.simpleEnergy(img)
#features  = hog(img, orientations=9, pixels_per_cell=(11, 11), cells_per_block=(1, 1), multichannel=True)
##features = 255 * features
#eHOG = e1.copy()
#cells = np.split(features, 9)
#cells = np.array(cells)
#max_bin = cells.max(axis=0)
#b=0
#nh = int(eHOG.shape[0])
#mh = int(eHOG.shape[1]) 
#for i in range(0, nh-11, 11):
#    for j in range(0, mh-11, 11):
#        eHOG[i:i+11, j:j+11]= e1[i:i+11, j:j+11] / max_bin[b]
#        b=b+1
img = SeamCarving.readImage("playa.jpg", 1)
hog = SeamCarving.eHOG(img)
representar_imagenes([eHOG], ["hog"])
#    for j in range(0, 9-2, 3):
#        print(i,":",i+3,",",j,":",j+3)

#x = np.arange(81)   
#a = np.split(x, 9)
#print(a)
#print(a[3])
#a = np.array(a)
#minimos = a.min(axis=1)
#print(minimos)
=======


mask = SeamCarving.readImage("mask.jpg", 0)
h, w = SeamCarving.maskSize(mask)
print(h,w)

















>>>>>>> Laura-DEV
