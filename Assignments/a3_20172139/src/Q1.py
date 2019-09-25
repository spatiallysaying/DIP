# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:58 2019

@author: E442282
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import e, pi
import math

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
 


'''
Fu= 1/N * SUM( Fx * e^(-j*2*pi*u*x)/N)

F(u,v) = SUM{ f(x,y)*exp(-j*2*pi*(u*x+v*y)/N) }


'''
  
def getFFT2D(gray):
    f = np.fft.fft2(gray)
    #DC component is at top-left corner
    #Therefore shift it to right&down so that DC is at center
    fshift = np.fft.fftshift(f)
    
#    %In Fourier transforms, high peaks are so high they 
#%hide details. Reduce contrast with the log function
    magnitude_spectrum = 20*np.log(1+np.abs(fshift))
#    magnitude_spectrum = 20*np.abs(fshift)
    return magnitude_spectrum
        


def fft1(x):
    N = len(x)
    if N <= 1: return x
    even = fft1(x[0::2])
    odd =  fft1(x[1::2])
    T= [np.exp(-2j*np.pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]


def fft(x):
    N = len(x)
    if N <= 1: return x
    Fu_even,Fu_odd = fft(x[0::2]),fft(x[1::2])
    W= [np.exp(-2j*np.pi*k/N)*Fu_odd[k] for k in range(N//2)]
    return [Fu_even[k] + W[k] for k in range(N//2)] + \
           [Fu_even[k] - W[k] for k in range(N//2)]
           

def fft2(gray):
    rows,cols=getImageDimnesion(gray)
    image = np.zeros((rows,cols))
    
    for row in range(rows):
        image[row, :] = fft(gray[row,:])
        
    for col in range(cols):
        image[:, col] = fft(image[:, col])    
    
    return image
  
print('\n\n=============\n\n')     
print('QUESTION # 1 :1. Implement 1D Fast Fourier Transform (Recursive Formulation).') 
print('1D Custom FFT verified against Numpy fft1')    
x = np.random.random(1024)           
print(np.allclose(fft1(x), np.fft.fft(x)))    
    
img = cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/lena.jpg')
plt.figure(figsize=(12, 12))
rgb,gray=getColorSpaces(img)


print('\n\nCustom FFT vs Numpy FFT')    
print(np.allclose(fft1(gray[0,:]), np.fft.fft(gray[0,:])))    
    

print('\n\n=============\n\n')      
#Compute custom 2D FFT
gray_fft= fft2(gray)
fshift = np.fft.fftshift(gray_fft)
magnitude_spectrum = 20*np.log(1+np.abs(fshift))


print('\n\nQUESTION # 1 :2 Use it to implement 2D FFT and display the result on suitable images of your choice.')  

print('\n2D Custom FFT verified against Numpy fft2') 

plt.subplot(1,3,1) 
plt.axis('off')
plt.title('Original') 
plt.imshow(gray,cmap='gray')
 
plt.subplot(1,3,2) 
plt.axis('off')
plt.title('Magnitude-Numpy ')
plt.imshow(getFFT2D(gray),cmap='gray')


plt.subplot(1,3,3) 
plt.axis('off')
plt.title('Magnitude-custom ')
plt.imshow(magnitude_spectrum,cmap='gray')




