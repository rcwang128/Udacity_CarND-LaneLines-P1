from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import cv2
import math
import os

os.listdir("test_images/")

#image = mpimg.imread('test_images/solidWhiteRight.jpg')
#image = mpimg.imread('test_images/solidYellowLeft.jpg')
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
#image = mpimg.imread('test_images/solidWhiteCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve2.jpg')


imshape = image.shape
print(type(image), imshape)
#plt.imshow(image)
#plt.show()

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_th, high_th):
    return cv2.Canny(img, low_th, high_th)

def blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Apply an image mask
def region_of_interest(img, vertices):
    mask = np.zeros_like(img) # A blank mask to start with

    if len(img.shape) > 2: # RGB
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else: # Gray
        ignore_mask_color = 255
    
    #filling pixels inside the polygon defined by vertices with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), \
            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Setting parameters
kernel_size = 5

low_th = 50
high_th = 150

vertices = np.array([[(160,imshape[0]), (410,330), (520,330), (900,imshape[0])]], dtype=np.int32) 

rho = 3
theta = np.pi/180
threshold = 1
min_line_length = 20
max_line_gap = 20


# Building a lane finding pipeline
image_gray = grayscale(image)
blur_gray = blur(image_gray, kernel_size)
edges = canny(blur_gray, low_th, high_th)
masked_edges = region_of_interest(edges, vertices) 
line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
color_edges = np.dstack((edges, edges, edges))
line_edges = cv2.addWeighted(color_edges, 0.8, line_img, 1, 0)

plt.subplot(141);plt.title('Original');plt.imshow(image)
plt.subplot(142);plt.title('Canny Edges');plt.imshow(edges)
plt.subplot(143);plt.title('Masked Edges');plt.imshow(masked_edges)
plt.subplot(144);plt.title('Lane Lines');plt.imshow(line_edges)
plt.show()








