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
    

#from scipy.signal import convolve2d 
#
#image1= cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/lena.jpg')
#rgb1,gray1=getColorSpaces(image1)
#f = cv2.resize(gray1, (256,256))
#
#image2= cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/bricks.jpg')
#rgb2,gray2=getColorSpaces(image2)
#h = cv2.resize(gray2, (256,256))
#
#
#f_Height, f_Width =  f.shape
#h_Height, h_Width = h.shape
#conv_fh = convolve2d(f,h)
#
#ff_conv = np.fft.fft2(conv_fh)
#ff_conv = np.absolute(ff_conv)
#
#plt.title("Spatial Convolve")
#plt.imshow(np.log(1 + np.abs(ff_conv)), cmap = 'gray')
#plt.show()
#f_pad = np.pad(f, [(0,h_Height-1), (0,h_Width-1)], mode='constant', constant_values=0)
#h_pad = np.pad(h, [(0,f_Height-1), (0,f_Width-1)], mode='constant', constant_values=0)
#F = np.fft.fft2(f_pad)
#H = np.fft.fft2(h_pad)
#
#mul_FH = np.multiply(F,H)
#plt.title("Frequency Domain Multiplication")
#plt.imshow(np.log(1 + np.abs(mul_FH)), cmap = 'gray')
#plt.show()
#mul_abs = np.absolute(mul_FH)
#
#delta = np.sum((mul_abs - ff_conv)**2)
#delta = delta/((h_Height + f_Height - 1) * (h_Width + f_Width - 1))
#print("Mean Squared error=" ,delta)




# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.
    
    High-frequency filters implemented:
        butterworth
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H
        
        .
    """

    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H = None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency 
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)
# End of class HomomorphicFilter


    

# Main code
img = cv2.imread(r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/test.tif',0)



ir,ic = img.shape
hr = ir//2;
hc = ic//2;
x,y =np.meshgrid(-hc:hc, -hr:hr); 

mg = sqrt((x/hc).^2 + (y/hr).^2);
lp = double(mg <= fc);
IM = fftshift(fft2(double(im0)));
IP = zeros(size(IM));
for z = 1:iz
    IP(:,:,z) = IM(:,:,z) .* lp;
end
im = abs(ifft2(ifftshift(IP)));









































