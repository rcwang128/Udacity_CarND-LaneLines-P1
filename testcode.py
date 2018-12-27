from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import cv2
import math
import os

os.listdir("test_images/")

#image = mpimg.imread('test_images/solidWhiteRight.jpg')
#image = mpimg.imread('test_images/solidYellowLeft.jpg')
#image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
#image = mpimg.imread('test_images/solidWhiteCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve.jpg')
image = mpimg.imread('test_images/solidYellowCurve2.jpg')

imshape = image.shape
print(type(image), imshape)
image_blank = np.zeros(imshape, dtype=np.uint8)
#plt.imshow(image_blank)
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

def draw_lines(img, lines, color=[255, 0, 0], thickness=7):
    slope_pos = np.array([])
    slope_neg = np.array([])

    intercept_y_pos = np.array([])
    intercept_y_neg = np.array([])    
   
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1) / (x2-x1) 
            intercept_y = y1 - slope * x1
            if slope > 0: # right lane
                #lane_right = np.append(lane_right, np.array([[slope, intercept_x, intercept_y]]),axis = 0)
                slope_pos = np.append(slope_pos, slope)
                intercept_y_pos = np.append(intercept_y_pos, intercept_y) 
            elif slope < 0: # left lane
                #lane_left = np.append(lane_left, np.array([[slope, intercept_x, intercept_y]]),axis = 0)
                slope_neg = np.append(slope_neg, slope)
                intercept_y_neg = np.append(intercept_y_neg, intercept_y)
    
    slope_right = slope_pos[abs(slope_pos - np.mean(slope_pos)) < 1.5*np.std(slope_pos)]
    intercept_right = intercept_y_pos[abs(intercept_y_pos - np.mean(intercept_y_pos)) < 1.5*np.std(intercept_y_pos)]
    
    slope_left = slope_neg[abs(slope_neg - np.mean(slope_neg)) < 1.5*np.std(slope_neg)]
    intercept_left = intercept_y_neg[abs(intercept_y_neg - np.mean(intercept_y_neg)) < 1.5*np.std(intercept_y_neg)]
    
    #print (round(slope_right,2), round(slope_left,2))
    #print (round(intercept_right,2), round(intercept_left,2))

    slope_right = round(np.mean(slope_right), 3)
    intercept_right = round(np.mean(intercept_right), 3)
    
    slope_left = round(np.mean(slope_left), 3)
    intercept_left = round(np.mean(intercept_left), 3)
    
    print (slope_right, intercept_right)
    print (slope_left, intercept_left)

    cv2.line(img, (520, int(520*slope_right+intercept_right)), (880, int(880*slope_right+intercept_right)), color, thickness)
    cv2.line(img, (120, int(120*slope_left+intercept_left)), (440, int(440*slope_left+intercept_left)), color, thickness)

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

vertices = np.array([[(120,imshape[0]), (420,330), (510,330), (900,imshape[0])]], dtype=np.int32) 

rho = 3
theta = np.pi/180
threshold = 1
min_line_length = 50
max_line_gap = 20 


# Building a lane finding pipeline
image_gray = grayscale(image)
blur_gray = blur(image_gray, kernel_size)
edges = canny(blur_gray, low_th, high_th)
masked_edges = region_of_interest(edges, vertices) 
line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
color_edges = np.dstack((edges, edges, edges))
line_edges = cv2.addWeighted(color_edges, 0.8, line_img, 1, 0)
line_edges = cv2.addWeighted(image, 0.8, line_img, 1, 0)

plt.subplot(141);plt.title('Original');plt.imshow(image)
plt.subplot(142);plt.title('Canny Edges');plt.imshow(edges)
plt.subplot(143);plt.title('Masked Edges');plt.imshow(masked_edges)
plt.subplot(144)
plt.title('Lane Lines');plt.imshow(line_edges)
#plt.imshow(image_blank)
plt.show()








