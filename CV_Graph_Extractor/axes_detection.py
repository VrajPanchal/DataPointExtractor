import numpy as np
import cv2

def detect_axes(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    horizontal_lines = []
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 45:
                horizontal_lines.append((length, line[0]))
            else:
                vertical_lines.append((length, line[0]))
    x_axis = max(horizontal_lines, key=lambda x: x[0])[1] if horizontal_lines else None
    y_axis = max(vertical_lines, key=lambda x: x[0])[1] if vertical_lines else None
    return x_axis, y_axis

def extend_axes(x_axis, y_axis):
    if x_axis is None or y_axis is None:
        return x_axis, y_axis
    x_slope = (x_axis[3] - x_axis[1]) / (x_axis[2] - x_axis[0]) if x_axis[2] != x_axis[0] else float('inf')
    y_slope = (y_axis[3] - y_axis[1]) / (y_axis[2] - y_axis[0]) if y_axis[2] != y_axis[0] else float('inf')
    if x_slope == y_slope:
        return x_axis, y_axis
    x_intercept = x_axis[1] - x_slope * x_axis[0]
    y_intercept = y_axis[1] - y_slope * y_axis[0]
    if x_slope == float('inf'):
        intersection_x = x_axis[0]
        intersection_y = y_slope * intersection_x + y_intercept
    elif y_slope == float('inf'):
        intersection_x = y_axis[0]
        intersection_y = x_slope * intersection_x + x_intercept
    else:
        intersection_x = (y_intercept - x_intercept) / (x_slope - y_slope)
        intersection_y = x_slope * intersection_x + x_intercept
    x_axis = [int(intersection_x), int(intersection_y), int(x_axis[2]), int(x_axis[3])]
    y_axis = [int(intersection_x), int(intersection_y), int(y_axis[2]), int(y_axis[3])]
    return x_axis, y_axis

def find_tickmarks(image, x_axis, y_axis):
    min_x_tickmark = (int(x_axis[0]), int(x_axis[1]))
    min_y_tickmark = (int(y_axis[0]), int(y_axis[1]))
    max_x_tickmark = (int(x_axis[2]), int(x_axis[3]))
    max_y_tickmark = (int(y_axis[2]), int(y_axis[3]))
    return min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark

def remove_axes_and_ticks(image, x_axis, y_axis, tick_length=10, line_thickness=3):
    cleaned_image = image.copy()
    cv2.line(cleaned_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), 255, line_thickness * 2)
    cv2.line(cleaned_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), 255, line_thickness * 2)
    for x in range(x_axis[0] - tick_length, x_axis[2] + tick_length + 1, tick_length):
        cv2.rectangle(cleaned_image, (x-3, x_axis[1] - 15), (x+3, x_axis[1] + 15), 255, -1)
    for y in range(y_axis[1] - tick_length, y_axis[3] + tick_length + 1, tick_length):
        cv2.rectangle(cleaned_image, (y_axis[0] - 15, y-3), (y_axis[0] + 15, y+3), 255, -1)
    return cleaned_image 