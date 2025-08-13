import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Provide the correct path to the image file
image_path = 'page_002_graph_2.png'

# Load the image in grayscale
image = cv.imread(image_path, 0)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image from path: {image_path}")
else:
    # Perform edge detection using Canny without Gaussian blur
    edges = cv.Canny(image, 50, 150)

    # Perform probabilistic Hough Line Transform
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

    # Convert the grayscale image to a BGR image so we can draw colored lines
    color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # Draw the lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color, thickness of 2

    # Display the original image with lines drawn on it
    plt.imshow(edges, cmap='gray')
    plt.show()

    # Optionally save the result
    output_image_path = '/Users/andyf/PycharmProjects/autoChart/Chart_Image_with_Edges.jpg'
    cv.imwrite(output_image_path, color_image)
