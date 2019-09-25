# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:58 2019

@author: E442282
"""

import numpy as np
import cv2
import os,sys
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import as_strided


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
    
    #In Fourier transforms, high peaks are so high they 
    #hide details. Reduce contrast with the log function
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


img = cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/land.png')
plt.figure(figsize=(12, 12))
rgb,gray=getColorSpaces(img)



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



img = cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/land.png')
plt.figure(figsize=(12, 12))
rgb,gray=getColorSpaces(img)



plt.subplot(2,2,1) 
plt.axis('off')
plt.title('Original') 
plt.imshow(gray,cmap='gray')
 
dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(dft)

magnitude_spectrum = getFFT2D(gray)

plt.subplot(2,2,2) 
plt.axis('off')
plt.title('FFT Magnitude ')
plt.imshow(magnitude_spectrum,cmap='gray')


F= np.fft.fft2(gray)

M,N=getImageDimnesion(gray)


# Set block around center of spectrum to zero
K = 20
magnitude_spectrum[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0

# Find all peaks higher than the 98th percentile
peaks = magnitude_spectrum < np.percentile(magnitude_spectrum, 98)

# Shift the peaks back to align with the original spectrum
peaks =  np.fft.ifftshift(peaks)

# Make a copy of the original (complex) spectrum
F_dim = F.copy()

# Set those peak coefficients to zero using peaks as mask
F_dim = F_dim * peaks.astype(int)

# Do the inverse Fourier transform to get back to an image.
# Since we started with a real image, we only look at the real part of
# the output.
image_filtered = np.real(np.fft.ifft2(F_dim))


magnitude_spectrum = np.log(1+np.abs(F_dim))

plt.subplot(2,2,3) 
plt.axis('off')
plt.title('Spectrum after suppression')
plt.imshow(magnitude_spectrum,cmap='gray')

plt.subplot(2,2,4) 
plt.axis('off')
plt.title('Reconstructed image');
plt.imshow(image_filtered,cmap='gray')
















