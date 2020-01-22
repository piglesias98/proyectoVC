# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:38:44 2020

@author: Paula
"""

import Basics
import energias
import SeamsCarving

img = Basics.readImage("imagenes/harry.jpg", 1)
mask_p = Basics.readImage("imagenes/positivo_harry.jpg", 0)
mask_n = Basics.readImage("imagenes/negativo_harry.jpg", 0)
#image = SeamsCarving.contentAwareResizing (img, img.shape[0] + 10, img.shape[1] + 10, False, energias.forwardEnergy)

#Basics.representar_imagenes([img, image], ["original", "editada"])

#image = Basics.readImage("arco.jpg", 1)
#image = SeamsCarving.contentAwareResizing (img, img.shape[0]-10, img.shape[1] + 10, True, energias.forwardEnergy)
image = SeamsCarving.objectRemoval(img, remove_mask=mask_n, preserve_mask=mask_p, rmask=True, pmask=True)

Basics.representar_imagenes([img, image], ["original", "editada (sin eficiencia)"])
