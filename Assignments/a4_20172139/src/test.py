import numpy as np
import cv2

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
  red, green, blue= cv2.split(img)
  
  return red, green, blue

def getBinaryImage(gray):
    ret,thresh= cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    return thresh
 
def cropBorder(image,dx=20,dy=20):
    h,w=getImageDimnesion(image)    
    im=image.copy()
    im = im[dy:h-dy,dx:w-dx]
    
    return im
    
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

#https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python
def getTopColors(im,num_colors=6):
    #Get color frequnecies
    color, colorcnt = np.unique(im.reshape(-1, im.shape[2]), axis=0, return_counts=True);
    #Sort the occarnces in reverse order to get top n-indices
    indices=(-colorcnt).argsort()[:6]
#    print(indices)
    
    bg_color=color[indices[0]]
    top_colors=[]
    #Most dominant color is background color,which is not text, threfore remove this item from the list
    # and consider only top 5 colors
    for idx in indices[1:]:
        top_colors.append(color[idx])
    
    return np.array(top_colors),bg_color

 
img_path=r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment4/input_data/fbpost.png'
    
#img_path=r'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment4/input_data/test_components.jpg'


img = cv2.imread(img_path)
img=cropBorder(img,20,20)

rgb,gray=getColorSpaces(img)

height,width=getImageDimnesion(img)

binary=getBinaryImage(gray)

plt.figure(figsize=(20, 20))

plt.subplot(3,2,1) 
plt.axis('off')
plt.title('Original-Crop border') 
plt.imshow(rgb)

im=rgb.copy()
top_colors,bg_color=getTopColors(im)
#print(top_colors)


#def getColoredText(im,color,bg_color):    
#    h,w = getImageDimnesion(img)
#    mask = im.copy()
#    
#    for i in range(h):
#        for j in range(w):
#            if not np.array_equal(im[i,j,:],color):
#                mask[i,j] = bg_color
#    return  mask           
#
#
#text_images = []
#for color in top_colors:
#    mask=getColoredText(im,color,bg_color)
#    text_images.append(mask)
#
#
#for color_img in text_images:
#    plt.figure()
#    plt.axis('off')
#    plt.imshow(color_img,cmap='gray')

def find(union_find,child,parent):
	if child in union_find:
		return find(union_find ,union_find[child],parent)
	else:
		return child
		
def ccl(binary):
    union_find = {}
    component_count = 1
    h,w = binary.shape
    binary = np.pad(binary,(0,1),'constant')
    img_labelled = np.zeros((h+1,w+1))
    for i in range(h):
        for j in range(w):
            if binary[i,j]==1:
                if i==0 and j==0:
                    img_labelled[i,j] = component_count
                elif i==0 and img_labelled[i,j-1]!=0:
                    img_labelled[i,j] = img_labelled[i,j-1]
                elif j==0 and img_labelled[i-1,j]!=0:
                    img_labelled[i,j] = img_labelled[i-1,j]
                
                elif img_labelled[i,j-1] + img_labelled[i-1,j-1] + img_labelled[i-1,j] + img_labelled[i-1,j+1] > 0:
                    aoi = np.array([img_labelled[i,j-1] , img_labelled[i-1,j-1] , img_labelled[i-1,j] , img_labelled[i-1,j+1]])
                    if np.sum(aoi==0) == 3:
                        img_labelled[i,j] = aoi[np.where(aoi!=0)]
                    else:
                        img_labelled[i,j] = min(aoi[np.where(aoi!=0)])
                    for k in aoi:
                        if k != 0 and k!= img_labelled[i,j]:
                            union_find[k] = img_labelled[i,j]        
                else:
                    component_count = component_count + 1
                    img_labelled[i,j] = component_count        
    for i in range(h):
        for j in range(w):
            if img_labelled[i,j]:
                img_labelled[i,j] = find(union_find,img_labelled[i,j],img_labelled[i,j])
                
    components = set()
    h,w = img_labelled.shape
    for i in range(h):
        for j in range(w):
            if img_labelled[i,j]:
                components.add(img_labelled[i,j])
    return img_labelled,components
    
    
def getColoredText(im,color,bg_color):    
    h,w = getImageDimnesion(img)
    mask = np.zeros_like(binary)
    
    for i in range(h):
        for j in range(w):
            if  np.array_equal(im[i,j,:],color):
                mask[i,j] = 1
    return  mask           


text_images = []
lst_labelled=[]

for color in top_colors:
    mask=getColoredText(im,color,bg_color)
    text_images.append(mask)

for color_img in text_images:
    plt.figure()
    plt.axis('off')    
    img_labelled,components=ccl(color_img)
    
    plt.title('No of Components '+str(len(components))) 
    plt.imshow(color_img)

 
def getObjectsWithHoles(binary):    
    image, contours, hierarchy  = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
    mask = np.zeros_like(img)
    img_copy=img.copy()    
    count=0
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]

        if currentHierarchy[2] >= 0:
            count=count+1
            cv2.drawContours(img_copy, [currentContour], -1, (255,255,0), 2)
            cv2.fillPoly(mask, pts =[currentContour], color=(255,255,0))
            mask = cv2.bitwise_and(img_copy, mask)
    
    print('Number of objects that have one or more holes:',count)        
    return  img_copy       



for color_img in text_images:
    plt.figure()
    plt.axis('off')    
    im=getObjectsWithHoles(color_img)

    plt.imshow(im)







 