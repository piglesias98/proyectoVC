# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:38:44 2020

@author: Paula
"""

import Basics
import SeamsCarving

img = Basics.readImage("imagenes/harry.jpg", 1)
mask_p = Basics.readImage("imagenes/positivo_harry.jpg", 0)
mask_n = Basics.readImage("imagenes/negativo_harry.jpg", 0)
#image = SeamsCarving.contentAwareResizing (img, img.shape[0] + 10, img.shape[1] + 10, False, energias.forwardEnergy)

#Basics.representar_imagenes([img, image], ["original", "editada"])

campo = Basics.readImage("imagenes/campo.jpg", 1)
#image = SeamsCarving.contentAwareResizing (campo, campo.shape[0]+40, campo.shape[1] + 10)

#image = SeamsCarving.objectRemoval(img, remove_mask=mask_n, rmask=True, preserve_mask=mask_p, pmask=True)
#
#Basics.representar_imagenes([img, image], ["original", "sin objeto"])

#Basics.representar_imagenes([campo, image], ["original", "enlarged"])
coche = Basics.readImage("imagenes/coches.jpg", 1)
mask_n = Basics.readImage("imagenes/mascara_coches1.jpg", 0)

eliminado = SeamsCarving.objectRemoval(coche, remove_mask=mask_n, rmask=True)
restaurado = SeamsCarving.contentAwareResizing(eliminado, coche.shape[0], coche.shape[1])

Basics.representar_imagenes([coche, eliminado, restaurado], ["Original", "Objeto eliminado", "Restaurado"])
