import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_text_in_region(original_image, bbox, index):
    x1, y1, x2, y2 = bbox
    roi = original_image[y1:y2, x1:x2]
    if len(roi.shape) == 2:
        roi_gray = roi
    else:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_adaptive_thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    roi_contrast = cv2.equalizeHist(roi_gray)
    #cv2.imwrite(f'debug_roi_{index}.png', roi)
    #cv2.imwrite(f'debug_roi_adaptive_thresh_{index}.png', roi_adaptive_thresh)
    #cv2.imwrite(f'debug_roi_contrast_{index}.png', roi_contrast)
    text = pytesseract.image_to_string(roi_adaptive_thresh, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789. -c tessedit_min_confidence=60')
    text = text.strip()
    data = pytesseract.image_to_data(roi_adaptive_thresh, output_type=pytesseract.Output.DICT)
    confidences = data['conf']
    return text, confidences, roi

def find_measure(image, start_x, start_y, direction):
    x, y = start_x, start_y
    while 0 <= x < image.shape[1] and image[y, x] < 128:
        x += direction[0]
    while 0 <= x < image.shape[1] and image[y, x] >= 128:
        x += direction[0]
    if x < 0 or x >= image.shape[1]:
        return None
    right_x = x
    top_y, bottom_y = y, y
    left_x = 0
    while left_x < right_x and any(image[top_y:bottom_y+1, left_x] < 128):
        left_x += 1
    while top_y > 0 and any(image[top_y-1, left_x:right_x+1] < 128):
        top_y -= 1
    while bottom_y < image.shape[0]-1 and any(image[bottom_y+1, left_x:right_x+1] < 128):
        bottom_y += 1
    for col in range(left_x, right_x):
        if any(image[top_y:bottom_y+1, col] < 128):
            left_x = col
            break
    left_x = max(0, left_x - 5)
    right_x = min(image.shape[1] - 1, right_x + 5)
    top_y = max(0, top_y - 5)
    bottom_y = min(image.shape[0] - 1, bottom_y + 5)
    return (left_x, top_y, right_x, bottom_y)

def find_x_measure(image, start_x):
    y = image.shape[0] - 1
    dark_pixel_count = 0
    while y >= 0:
        if any(image[y, max(0, start_x-15):min(image.shape[1], start_x+16)] < 100):
            dark_pixel_count += 1
            if dark_pixel_count >= 3:
                break
        else:
            dark_pixel_count = 0
        y -= 1
    if y < 0:
        return None
    dark_pixel_y = y
    white_pixel_count = 0
    while y >= 0:
        if all(image[y, max(0, start_x-15):min(image.shape[1], start_x+16)] > 200):
            white_pixel_count += 1
            if white_pixel_count >= 3:
                break
        else:
            white_pixel_count = 0
        y -= 1
    if y < 0:
        return None
    white_pixel_y = y
    mask = np.zeros((image.shape[0]+2, image.shape[1]+2), np.uint8)
    flood_fill_image = image.copy()
    flood_region = image[max(0, dark_pixel_y-50):min(image.shape[0], dark_pixel_y+50), max(0, start_x-100):min(image.shape[1], start_x+100)]
    flood_mask = np.zeros((flood_region.shape[0]+2, flood_region.shape[1]+2), np.uint8)
    cv2.floodFill(flood_region, flood_mask, (min(15, flood_region.shape[1]-1), min(50, flood_region.shape[0]-1)), 128, loDiff=30, upDiff=30)
    filled_region = np.where(flood_region == 128)
    if len(filled_region[0]) == 0:
        return None
    left_x = max(0, start_x-100) + filled_region[1].min()
    right_x = max(0, start_x-100) + filled_region[1].max()
    top_y = white_pixel_y + 1
    bottom_y = dark_pixel_y
    while left_x < right_x and np.all(image[top_y:bottom_y+1, left_x] >= 128):
        left_x += 1
    while right_x > left_x and np.all(image[top_y:bottom_y+1, right_x] >= 128):
        right_x -= 1
    while top_y < bottom_y and np.all(image[top_y, left_x:right_x+1] >= 128):
        top_y += 1
    while bottom_y > top_y and np.all(image[bottom_y, left_x:right_x+1] >= 128):
        bottom_y -= 1
    while left_x > 0 and np.any(image[top_y:bottom_y+1, left_x] < 128):
        left_x -= 1
    while right_x < image.shape[1] - 1 and np.any(image[top_y:bottom_y+1, right_x] < 128):
        right_x += 1
    while top_y > 0 and np.any(image[top_y, left_x:right_x+1] < 128):
        top_y -= 1
    while bottom_y < image.shape[0] - 1 and np.any(image[bottom_y, left_x:right_x+1] < 128):
        bottom_y += 1
    left_x = max(0, left_x - 5)
    right_x = min(image.shape[1] - 1, right_x + 5)
    top_y = max(0, top_y - 5)
    bottom_y = min(image.shape[0] - 1, bottom_y + 5)
    return (left_x, top_y, right_x, bottom_y)

def find_measures(image, x_axis, y_axis):
    measures = []
    y_max = find_measure(image, y_axis[2], y_axis[3], (-1, 0))
    y_min = find_measure(image, y_axis[0], y_axis[1], (-1, 0))
    if y_max:
        measures.append(tuple(map(int, y_max)))
    if y_min:
        measures.append(tuple(map(int, y_min)))
    min_x_measure = find_x_measure(image, x_axis[0])
    max_x_measure = find_x_measure(image, x_axis[2])
    if min_x_measure:
        measures.append(tuple(map(int, min_x_measure)))
    if max_x_measure:
        measures.append(tuple(map(int, max_x_measure)))
    return measures 