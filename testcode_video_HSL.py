from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#os.listdir("test_images/")

#global slope_right_next
#global intercept_right_next
#global slope_left_next
#global intercept_left_next

(slope_right_next, intercept_right_next) = (0.69, -30)
(slope_left_next, intercept_left_next) = (-0.71, 650)


def HSL_conversion(img):
    hsl_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # BGR format in openCV
    lower_yellow = np.array([0,60,100], dtype=np.uint8)
    upper_yellow = np.array([110,255,255], dtype=np.uint8)
    lower_white = np.array([10,180,20], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    yellow = cv2.inRange(hsl_image, lower_yellow, upper_yellow)
    white = cv2.inRange(hsl_image, lower_white, upper_white)
    mask_yellow_n_white = cv2.bitwise_or(yellow, white)
    return cv2.bitwise_and(img, img, mask = mask_yellow_n_white)

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
    
    global slope_right_next
    global intercept_right_next
    global slope_left_next
    global intercept_left_next
   
    for line in lines:
        x = []
        y = []
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), [0,255,0], 2) 
            #slope = (y2-y1) / (x2-x1) 
            #intercept_y = y1 - slope * x1
            #print(x1,y1,x2,y2)
            x += [x1,x2]
            y += [y1,y2]
            (slope, intercept_y) = np.polyfit(x,y,1)
            if slope > 0: # right lane
                #print(round(slope,3))
                #lane_right = np.append(lane_right, np.array([[slope, intercept_x, intercept_y]]),axis = 0)
                slope_pos = np.append(slope_pos, round(slope,2))
                intercept_y_pos = np.append(intercept_y_pos, round(intercept_y,2))
                #print(slope_pos)
            elif slope < 0: # left lane
                #lane_left = np.append(lane_left, np.array([[slope, intercept_x, intercept_y]]),axis = 0)
                slope_neg = np.append(slope_neg, round(slope,2))
                intercept_y_neg = np.append(intercept_y_neg, round(intercept_y,2))
    
    #print(slope_pos)
    #print(intercept_y_pos)
    
    if (len(slope_pos)>2 and len(intercept_y_pos)>2): 
        slope_right = slope_pos[abs(slope_pos - np.mean(slope_pos)) < np.std(slope_pos)]
        intercept_right = intercept_y_pos[abs(intercept_y_pos - np.mean(intercept_y_pos)) < np.std(intercept_y_pos)]
        
        slope_right = round(np.mean(slope_right), 2)
        intercept_right = round(np.mean(intercept_right), 2)
        
        slope_right_next = slope_right
        intercept_right_next = intercept_right
        #print(slope_right, intercept_right)

    elif (len(slope_pos)>0 and len(intercept_y_pos)>0):
        slope_right = np.mean(slope_pos)
        intercept_right = np.mean(intercept_y_pos)
        
        slope_right_next = slope_right
        intercept_right_next = intercept_right
        #print(slope_right, intercept_right)

    else:
        slope_right = slope_right_next
        intercept_right = intercept_right_next
        #print(slope_right, intercept_right)

    if (len(slope_neg)>2 and len(intercept_y_neg)>2): 
        slope_left = slope_neg[abs(slope_neg - np.mean(slope_neg)) < np.std(slope_neg)]
        intercept_left = intercept_y_neg[abs(intercept_y_neg - np.mean(intercept_y_neg)) < np.std(intercept_y_neg)]
        
        slope_left = round(np.mean(slope_left), 2)
        intercept_left = round(np.mean(intercept_left), 2)
        
        slope_left_next = slope_left
        intercept_left_next = intercept_left
        #print(slope_left, intercept_left)

    elif (len(slope_neg)>0 and len(intercept_y_neg)>0):
        slope_left = np.mean(slope_neg)
        intercept_left = np.mean(intercept_y_neg)
        
        slope_left_next = slope_left
        intercept_left_next = intercept_left
        #print(slope_left, intercept_left)

    else:
        slope_left = slope_left_next
        intercept_left = intercept_left_next
        #print(slope_left, intercept_left)
    

    #print(slope_left, intercept_left)
    #print(slope_right, intercept_right)
    if(~np.isnan(slope_left) and ~np.isnan(intercept_left) and ~np.isnan(slope_right) and ~np.isnan(intercept_right)):
        cv2.line(img, (520, int(520*slope_right+intercept_right)), (880, int(880*slope_right+intercept_right)), color, thickness)
        cv2.line(img, (120, int(120*slope_left+intercept_left)), (440, int(440*slope_left+intercept_left)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), \
            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def process_image(image):
    imshape = image.shape 
    # Setting parameters
    kernel_size = 5
    low_th = 50
    high_th = 150
    vertices = np.array([[(120,imshape[0]), (420,330), (510,330), (900,imshape[0])]], dtype=np.int32) 
    #vertices = np.array([[(0,imshape[0]), (420,300), (500,300), (imshape[1],imshape[0])]], dtype=np.int32) 
    rho = 3
    theta = np.pi/180
    threshold = 1
    min_line_length = 60 #50
    max_line_gap = 30 #20 
    
    # Building a lane finding pipeline
    image_hsl = HSL_conversion(image)
    image_gray = grayscale(image_hsl)
    blur_gray = blur(image_gray, kernel_size)
    edges = canny(blur_gray, low_th, high_th)
    masked_edges = region_of_interest(edges, vertices) 
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    color_edges = np.dstack((edges, edges, edges))
    line_edges = cv2.addWeighted(image, 0.8, line_img, 1, 0)
    return line_edges

white_output = 'test_videos_output/solidWhiteRight.mp4'
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
challenge_output = 'test_videos_output/challenge.mp4'

clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4")#.subclip(0,8)
clip3 = VideoFileClip("test_videos/challenge.mp4")

white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

#challenge_clip = clip3.fl_image(process_image)
#challenge_clip.write_videofile(challenge_output, audio=False) 
