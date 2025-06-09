import cv2
import numpy as np
import os
img = cv2.imread('Output_2008\extracted_graphs\page_001_graph_1.png')

lines_list = list()

if len(img.shape) == 3: 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_all =  img.copy()
else:
    gray = img.copy()
    image_all = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

edges = cv2.Canny(gray,50,150,apertureSize=3)

lines = cv2.HoughLinesP(
        edges, # Input edge image
        1, # Distance resolution in pixels
        np.pi / 180, # Angle resolution in radians
        100, # Min number of votes for valid line
        minLineLength = 100, # Min allowed length of line
        maxLineGap = 10 # Max allowed gap between line for joining them
        )


for points in lines:
    # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points On the original image
    cv2.line(image_all,(x1,y1),(x2,y2),(0,100,255),2)

#make dirk
if not os.path.exists('Output_2008\extracted_lines'):
    os.makedirs('Output_2008\extracted_lines')

cv2.imwrite('Output_2008\extracted_lines\line_extracted.png',image_all)