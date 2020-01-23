# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:38:44 2020

@author: Paula
"""

import Basics
import SeamsCarving

'''
Leemos las imágenes de prueba
'''
campo = Basics.readImage("imagenes/campo.jpg", 1)

arbol1 = Basics.readImage("imagenes/arbol1.jpg", 1)
arbol2 = Basics.readImage("imagenes/arbol2.jpg", 1)

arco1 = Basics.readImage("imagenes/arco1.jpg", 1)
arco2 = Basics.readImage("imagenes/arco2.jpg", 1)

hombre = Basics.readImage("imagenes/hombre.jpg", 1)

roca = Basics.readImage("imagenes/roca.jpg", 1)

harry = Basics.readImage("imagenes/harry.jpg", 1)
harry_p = Basics.readImage("imagenes/positivo_harry.jpg", 0)
harry_n = Basics.readImage("imagenes/negativo_harry.jpg", 0)

coche = Basics.readImage("imagenes/coches.jpg", 1)
coche_n1 = Basics.readImage("imagenes/mascara_coches1.jpg", 0)
coche_n2 = Basics.readImage("imagenes/mascara_coches2.jpg", 0)


'''
Content Aware Resizing
'''





#image = SeamsCarving.contentAwareResizing (img, img.shape[0] + 10, img.shape[1] + 10, False, energias.forwardEnergy)

#Basics.representar_imagenes([img, image], ["original", "editada"])


#image = SeamsCarving.contentAwareResizing (campo, campo.shape[0]+40, campo.shape[1] + 10)

#image = SeamsCarving.objectRemoval(img, remove_mask=mask_n, rmask=True, preserve_mask=mask_p, pmask=True)
#
#Basics.representar_imagenes([img, image], ["original", "sin objeto"])

#Basics.representar_imagenes([campo, image], ["original", "enlarged"])



eliminado = SeamsCarving.objectRemoval(coche, remove_mask=mask_n, rmask=True)
restaurado = SeamsCarving.contentAwareResizing(eliminado, coche.shape[0], coche.shape[1])

Basics.representar_imagenes([coche, eliminado, restaurado], ["Original", "Objeto eliminado", "Restaurado"])
