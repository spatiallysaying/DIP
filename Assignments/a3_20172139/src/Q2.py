# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:58 2019

@author: E442282
"""

import numpy as np
import cv2
import os,sys
from matplotlib import pyplot as plt


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
    magnitude_spectrum = 20*np.log(1+np.abs(fshift))
#    magnitude_spectrum = 20*np.abs(fshift)
    return magnitude_spectrum
            

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

 
def applyIdealFilter(gray,D0):
    
    M,N=getImageDimnesion(gray)
    F=np.fft.fft2(gray)
    u=np.arange(M)
    v=np.arange(N)
    idimg=np.where(u>M//2)    
    u[idimg]=u[idimg]-M
    idy=np.where(v>N//2)
    v[idy]=v[idy]-N
    V,U=np.meshgrid(v,u)

    Duv=np.sqrt(np.power(U, 2) +np.power(V, 2))    
    H = (Duv<=D0).astype(int)
    
    G=np.multiply(H,F) 
    g=np.real(np.fft.ifft2((G)))
    
    return g

def applyButterworthFilter(gray,D0,filter_order):
    
    M,N=getImageDimnesion(gray)
    F=np.fft.fft2(gray)
    u=np.arange(M)
    v=np.arange(N)
    idimg=np.where(u>M//2)    
    u[idimg]=u[idimg]-M
    idy=np.where(v>N//2)
    v[idy]=v[idy]-N
    V,U=np.meshgrid(v,u)  
    
    Duv=np.sqrt(np.power(U, 2) +np.power(V, 2))  
    H = np.power( 1/(1 + Duv/D0) ,2*filter_order)
    
    G=np.multiply(H,F) 
    g=np.real(np.fft.ifft2((G)))
    
    return g

def applyGaussianFilter_Freq(gray,D0,filter_order):
    
    M,N=getImageDimnesion(gray)
    F=np.fft.fft2(gray)
    u=np.arange(M)
    v=np.arange(N)
    idimg=np.where(u>M//2)    
    u[idimg]=u[idimg]-M
    idy=np.where(v>N//2)
    v[idy]=v[idy]-N
    V,U=np.meshgrid(v,u)  
    
    Duv=np.sqrt(np.power(U, 2) +np.power(V, 2))  
        
    H = (np.exp(-(np.power(Duv,2))/(2*(np.power(D0,2)))))
    
    G=np.multiply(H,F) 
    g=np.real(np.fft.ifft2((G)))
    
    return g


img = cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/lena.jpg')
rgb,gray=getColorSpaces(img)


plt.axis('off')
plt.title('Original Image')
plt.imshow(gray,cmap='gray')
plt.show()

print('\n==========\n')
print('\nQUESTION 2 :1. Implement the Ideal, Butterworth and Gaussian Low Pass Filters and apply them on lena.jpg.')

print("-----ILPF---")
plt.figure(figsize=(12, 12))
lstCutoffFreq=[10,30,60,160,460]
num=0
for cutoff in lstCutoffFreq:    
    plt.subplot(1,5,num+1) 
    plt.axis('off')
    plt.title('Cutoff Freq =' +str(cutoff))      
    plt.imshow(applyIdealFilter(gray,cutoff),cmap='gray')
    num=num+1
 
plt.show()
plt.figure(figsize=(12, 12))
num=0    
print("-----BLPF---")    
for cutoff in lstCutoffFreq:    
    plt.subplot(1,5,num+1) 
    plt.axis('off')
    plt.title('D0 =' +str(cutoff) + 'n=2')      
    plt.imshow(applyButterworthFilter(gray,cutoff,2),cmap='gray')
    num=num+1

plt.show()

num=0    
plt.figure(figsize=(12, 12))
print("-----GLPF---")    
for cutoff in lstCutoffFreq:    
    plt.subplot(1,5,num+1) 
    plt.axis('off')
    plt.title('D0 =' +str(cutoff) + ' n=2')      
    plt.imshow(applyGaussianFilter_Freq(gray,cutoff,2),cmap='gray')
    num=num+1
plt.show()    


plt.figure(figsize=(4, 4))
print('\n==========')
print('\nQUESTION 2 :2. Using lena.jpg, apply the GLPF with two different values of Compute \
      the difference of the two outputs and display it. Report your observations')

low_blur=applyGaussianFilter_Freq(gray,60,2)
high_blur=applyGaussianFilter_Freq(gray,10,2)

diff=high_blur-low_blur
plt.axis('off')
plt.title('Gaussian sigma1-sigma2')
plt.imshow(getNormalizedImage(diff),cmap='gray')

'''
 Subtracting one image from the other preserved spatial information that lies between 
 the range of frequencies that are preserved in the two blurred images.
 Thus, the difference of Gaussians is a band-pass filter that discards all but a 
 handful of spatial frequencies  that are present in the original grayscale image
 
 As the difference between two differently low-pass filtered images, the DoG is actually 
 a band-pass filter, which removes high frequency components representing noise, and also
 some low frequency components representing the homogeneous areas in the image. 
 The frequency  components in the passing band are assumed to be associated to the
 edges in the images.
 
 Properties
 Detect edges in the image
 Bandpass filter
 
 
 
 
 '''

'''
attenuate a selected frequency (and some of its neighbors) and leave other 
frequencies of the Fourier transform relatively unchanged

Repetitive noise in an image is sometimes seen as a bright peak somewhere
 other than the origin. You can suppress such noise effectively by carefully 
 erasing the peaks. One way to do this is to use a notch filter to simply 
 remove that frequency from the picture. 
 
 Although it is possible to create notch filters for common noise patterns,
 in general notch filtering is an ad hoc procedure requiring a human expert 
 to determine what frequencies need to be removed to clean up the signal.
'''














