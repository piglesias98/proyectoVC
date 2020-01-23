# -*- coding: utf-8 -*-

import numpy as np

import Basics
import energias

'''
Función para modificar el tamaño de una imagen

Entrada:
    -> img: imagen
    -> nn: nuevo número de filas
    -> nm: nuevo número de columnas
    -> efficiency: True-> utiliza el "scale and Carve"
    -> energy: función de energía que se va a utilizar

Salida:
    -> imagen con tamaño modificado
'''
def contentAwareResizing (img, nn, nm, efficiency=True, energy=energias.forwardEnergy, draw=False):

    n, m = img.shape[:2]

    dif_n = n - nn
    dif_m = m - nm

    accion = [Basics.removeOrderSeams, Basics.addOrderSeams]
    funcion = Basics.scaleAndCarve

    if efficiency == False:
        funcion = Basics.carve

    if dif_n > 0:

        if dif_m < 0:
            img = funcion(img, nn, m, accion[0], energy, draw)
            img = funcion(img, nn, nm, accion[1], energy, draw)

        else:
            img = funcion(img, nn, nm, accion[0], energy, draw)

    elif dif_n < 0:

        if dif_m > 0:
            img = funcion(img, n, nm, accion[0], energy, draw)
            img = funcion(img, nn, nm, accion[1], energy, draw)

        else:
            img = funcion(img, nn, nm, accion[1], energy, draw)

    else:

        if dif_m < 0:
            img = funcion(img, nn, nm, accion[1], energy, draw)

        else:
            img = funcion(img, n, nm, accion[0], energy, draw)

    return img 

'''
Función para eliminar o conservar un objeto de una imagen

Entrada:
    -> image: imagen
    -> remove_mask: máscara de la imagen para el elemento que se quiere eliminar
    -> preserve_mask: máscara de la imagen para el elemento que se quiere conservar
    -> nn: número de filas a eliminar
    -> nm: número de columnas a eliminar
    -> rmask: False -> no se usa el parámetro remove_mask
              True -> se usa el parámetro remove_mask
    -> pmask: False -> no se usa el parámetro remove_mask
              True -> se usa el parámetro remove_mask

Salida:
    -> Imagen con el orbjeto eliminado o in modificar
'''
def objectRemoval(image, remove_mask=None, preserve_mask=None, nn=0, nm=0, rmask=False, pmask=False):
    img = image.copy()
    n,m = image.shape[:2]

    if rmask:
        nn, nm = Basics.maskSize(remove_mask)

    if nn < nm:   #Eliminamos las filas

        print("\nNumero de seams horizontales a eliminar: ", nn)
        #Rotamos
        img = np.rot90(img, k=-1, axes=(0, 1))

        if rmask:
            remove_mask = np.rot90(remove_mask, k=-1, axes=(0, 1))

        if pmask:
            preserve_mask = np.rot90(preserve_mask, k=-1, axes=(0, 1))


        #Eliminamos las horizontales que sobren
        for i in range(abs(nn)):
            print("Eliminamos ", i)
            a, b, path = Basics.verticalSeam(img, energias.forwardEnergy, remove_mask, preserve_mask, rmask, pmask)
            img = Basics.removeSeam(img, path)


            if rmask:
                remove_mask = Basics.removeSeam(remove_mask, path)

            if pmask:
                preserve_mask = Basics.removeSeam(preserve_mask, path)


        #Rotamos
        return np.rot90(img, k=1, axes=(0, 1))
#        return img
    else:

        print("\nNumero de seams verticales a eliminar: ", abs(nm))
        #Eliminamos las verticales que sobren
        for i in range(abs(nm)):
            print("Eliminamos", i)
            a, b, path = Basics.verticalSeam(img, energias.forwardEnergy, remove_mask, preserve_mask, rmask, pmask)
            img = Basics.removeSeam(img, path)

            if rmask:
                remove_mask = Basics.removeSeam(remove_mask, path)

            if pmask:
                preserve_mask = Basics.removeSeam(preserve_mask, path)

        return img
