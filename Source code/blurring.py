import cv2
import random

# read in an image
img = cv2.imread("lena.tif")
cv2.imshow("Lena",img)

# blur the image using gaussian blur
gblur = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT) # cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType=BORDER_DEFAULT]]] )
cv2.imshow('Gaussian Blur', gblur)



# https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/

# salt and pepper
def salt_and_pepper(img):
 
    # Getting the dimensions of the image
    row , col = img.shape
     
    # Randomly pick some pixels in the image for coloring them white
    # Pick a random number between 300 and 10000 (can be any value)
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        img[y_coord][x_coord] = 255
         
    # Randomly pick some pixels in the image for coloring them black
    # Pick a random number between 300 and 10000 (can be any value)
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        img[y_coord][x_coord] = 0
         
    return img
 
# salt-and-pepper noise can be applied only to grayscale images
# Convert the colour image into grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
 
#Storing the image
cv2.imshow('salt and pepper', salt_and_pepper(gray))

cv2.waitKey(0)