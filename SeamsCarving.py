# -*- coding: utf-8 -*-

import cv2
import numpy as np

import Basics
import energias



def objectRemoval(image, remove_mask=None, preserve_mask=None, nn=0, nm=0):
    
    energy = energias.forwardEnergy(image)
    
    if remove_mask != None:
        nn, nm = Basics.maskSize(remove_mask)
        energy = Basics.removeEnergy(energy, remove_mask)
        
    if preserve_mask != None:
        energy = Basics.removeEnergy(energy, preserve_mask)
    
    return contentAwareResizing(image, nn, nm, True,energy)
        
