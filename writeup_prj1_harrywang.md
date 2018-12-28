# **Finding Lane Lines on the Road** 
# Project 1
# Harry Wang
---


### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted several steps.

Step 1
Convert the images into HSL (hue, saturation, value) domain for better color recognition.
Yellow color and white color ranges are defined to detect the lines on the road.

Step 2
Convert the images to grayscale for blurring.

Step 3
Apply Gaussian blur function to smooth the image.

Step 4
Apply Canny edge detection function to find all edges in the image.

Step 5
Define a region of interest (camera facing area) and only keep the detected edges in that region.

Step 6
Apply Hough line transform to find all lines based on detected edges.

Step 7
Within step 6 function, lines are averaged/extrapolated and drawn on the picture.

Inside draw_lines() function, I've calculated the slope and y-axis intercept point for all detected lines during a frame. And separates the data sets based on the slope polarity, i.e. positive slopes for the right lane and negative slopes for the left lane. For each set, I got rid of the data with large standard deviation, which could potentially be incorrect detected lines. For remaining slopes and intercept points, I averaged them and then draw one left and one right lines. The lines are extrapolated based on given coordinates. Additionally, since some frames may not have lines correctly detected, I would then use the (slope, intercept) data from previous frame.

Step 8
Drawn lines are weighted and added together with source images. 

[image1]: ./test_images_output/matplots.png
[image2]: ./test_images_output/solidwhiteright.png
[image3]: ./test_images_output/solidyellowleft.png
[image4]: ./test_images_output/solidcarlaneswitch.png
[image5]: ./test_images_output/solidwhitecurve.png
[image6]: ./test_images_output/solidyellowcurve.png
[image7]: ./test_images_output/solidyellowcurve2.png

All videos are under ./test_images_output/


### 2. Identify potential shortcomings with your current pipeline

One shortcoming would be the averaged data sets (slope and intercept point) are still very jittery. My algorithm is not good enough to smooth them. It is partially because sometimes there is no lines detected, or no data left after filtering (using standard deviation). I had to use a lot of calculation to manipulate the data sets so that the code could be executed properly. 

Another shortcoming would be the lines were not able to be detected when curved. Fitting curved lanes was unsuccessful.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be updating the filtering algorithm to better locate the averaged line info (slope and intercept points for example) for each frame.

Another potential improvement could be to implement an algorithm to better correlate the data frame by frame so that the extrapolated lines would be more stable.

