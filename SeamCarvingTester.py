# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:38:44 2020

@author: Paula
"""

import Basics
import Energias
import SeamCarving

img = Basics.readImage("arco.jpg", 1)
#image = SeamCarving.contentAwareResizing (img, img.shape[0] + 5, img.shape[1] + 10, False, Energias.forwardEnergy)

#Basics.representar_imagenes([img, image], ["original", "editada"])

image = Basics.readImage("arco.jpg", 1)
image1 = SeamCarving.contentAwareResizing (img, img.shape[0] - 10, img.shape[1] - 5, True, Energias.forwardEnergy)

Basics.representar_imagenes([image, image1], ["Sin eficiencia", "con eficiencia"])
