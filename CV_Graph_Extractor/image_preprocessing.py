import numpy as np
import cv2

def remove_pixels(image):
    mask = np.logical_and(image >= 80, image <= 255)
    image[mask] = 255
    return image

def flood_fill(image, x, y):
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    flood_fill_image = image.copy()
    cv2.floodFill(flood_fill_image, mask, (x,y), 255, loDiff=5, upDiff=5)
    return cv2.bitwise_not(flood_fill_image - image) 