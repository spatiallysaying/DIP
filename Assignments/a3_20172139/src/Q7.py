# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:58 2019

@author: E442282
"""

import numpy as np
import cv2
import os,sys
from matplotlib import pyplot as plt
import scipy
from PIL import Image
import time
from skimage.measure import block_reduce



def getColorSpaces(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    return rgb,gray

def getImageDimnesion(image):
    height,width = image.shape[:2]
    
    return height,width

def showImage(image,title,cmap):
    plt.imshow(image,cmap=cmap)
    plt.axis('off')
    plt.title(title)


def splitRGBChannels(image):
  red, green, blue= cv2.split(image)
  
  return red, green, blue
 
    
def getNormalizedImage(image):
    norm_image=image.copy()
    norm_image = np.maximum(norm_image, np.zeros(norm_image.shape))
    norm_image = np.minimum(norm_image, 255 * np.ones(norm_image.shape))
    norm_image = norm_image.round().astype(np.uint8)
    
    return norm_image
 

def getFFT2D(gray):
    f = np.fft.fft2(gray)
    #DC component is at top-left corner
    #Therefore shift it to right&down so that DC is at center
    fshift = np.fft.fftshift(f)
    
#    %In Fourier transforms, high peaks are so high they 
#%hide details. Reduce contrast with the log function
    magnitude_spectrum = np.log(1+np.abs(fshift))
#    magnitude_spectrum = 20*np.abs(fshift)
    return magnitude_spectrum
            
#def getFFT2D_2(gray):
#    f = np.fft.fft2(gray)
#    fshift = np.fft.fftshift(f)
#    magnitude_spectrum = np.log(1+np.abs(fshift))
#    res =  np.fft.ifft2( np.fft.ifftshift(f)).astype("uint8")
#    return res



def get_ifft_img(img):
    # Take 2d fft
    f = np.fft.ifft2(img)
    # Apply fft shift for visualization
    fshift = np.fft.ifftshift(f)
    # We are only checking the magnitude 
    # We use log as transformation, hence values should be > 1
    magnitude_spectrum = np.log(np.abs(fshift) +1)
    return magnitude_spectrum


def getRotatedImage(gray,angle):
    h,w=getImageDimnesion(gray)
    center = (w / 2, h / 2)    
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(gray, M, (w, h))
    
    return rotated


def getTranslatedImage(gray,xShift,yShift):
    h,w=getImageDimnesion(gray)
    
    M = np.float32([[1,0,xShift],[0,1,yShift]])
    translated = cv2.warpAffine(gray,M,(w,h))

    
    return translated
    

def getSampledImage(im,nx,ny):
    dim=(nx,ny)
#    return cv2.resize(im, dim)
    return block_reduce(im, dim)
    
    
def displayImagepair(src,src_title,target,target_title):
    plt.subplot(1,2,1) 
    plt.axis('off')
    plt.title(src_title) 
    plt.imshow(src,cmap='gray')
     
    plt.subplot(1,2,2) 
    plt.axis('off')
    plt.title(target_title)
    plt.imshow(target,cmap='gray')
    plt.show()


image= cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/bricks.jpg')
rgb,gray=getColorSpaces(image)
#gray=cv2.resize(gray,(256,256))
#gray =  cv2.GaussianBlur(gray, (11,11),0) 

# Show images and their FFT
for i in range(1,16):    
    sampled_image = getSampledImage(gray,i,i)
    F = getFFT2D(sampled_image)
    src_title="nx:{} ny:{}".format(i,i)
    target_title='FFT'
    displayImagepair(sampled_image,src_title,F,target_title)
    
    
  
#nx=2 and ny=2

#Gaussian blur
#nx=4 and ny=4






   
       

