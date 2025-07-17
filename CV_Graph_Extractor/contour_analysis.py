import numpy as np
import cv2
from measure_detection import get_text_in_region

def create_debug_image(curves, x_axis, y_axis, white_background=True, original_image=None):
    if white_background:
        debug_image = np.ones((max(y_axis[1], y_axis[3]), max(x_axis[0], x_axis[2]), 3), dtype=np.uint8) * 255
    else:
        debug_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    # Draw axes in red
    cv2.line(debug_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
    cv2.line(debug_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 2)
    
    # Draw curves
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Add more colors if needed
    for i, curve in enumerate(curves):
        color = colors[i % len(colors)]
        curve_np = np.array(curve, dtype=np.int32)
        cv2.polylines(debug_image, [curve_np], False, color, 1)
    
    # Draw max and min points
    cv2.circle(debug_image, (x_axis[0], x_axis[1]), 5, (0, 255, 0), -1)
    cv2.circle(debug_image, (x_axis[2], x_axis[3]), 5, (0, 255, 0), -1)
    cv2.circle(debug_image, (y_axis[0], y_axis[1]), 5, (0, 255, 0), -1)
    cv2.circle(debug_image, (y_axis[2], y_axis[3]), 5, (0, 255, 0), -1)
    
    return debug_image

def create_white_background_image(cleaned_image, result_image, x_axis, y_axis, filtered_contours, 
                                  min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark,
                                  measures,original_image):
    # Create a white image of the same size as the original
    white_background = np.ones_like(result_image) * 255

    # Define the bounding rectangle
    min_x, max_x = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    min_y, max_y = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])

    # Draw the axes in red
    cv2.line(white_background, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
    cv2.line(white_background, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 2)

    # Draw the filtered contours in green
    for curve in filtered_contours:
        curve_np = np.array(curve, dtype=np.int32)
        cv2.polylines(white_background[min_y:max_y, min_x:max_x], [curve_np], False, (0, 255, 0), 2)

    # Function to add text with a white background
    def put_text_with_background(img, text, position, font_scale=0.9, thickness=2, text_color=(255,0,0), bg_color=(255,255,255)):
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_offset_x, text_offset_y = position
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    # Draw green dots and value labels for max and min locations
    for i, measure in enumerate(measures):
        text, _, _ = get_text_in_region(original_image, measure, i)
        if i == 0:  # max Y
            cv2.circle(white_background, (max_y_tickmark[0], max_y_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(white_background, text, (max_y_tickmark[0]-70, max_y_tickmark[1]+10))
        elif i == 1:  # min Y
            cv2.circle(white_background, (min_y_tickmark[0], min_y_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(white_background, text, (min_y_tickmark[0]-70, min_y_tickmark[1]+10))
        elif i == 2:  # min X
            cv2.circle(white_background, (min_x_tickmark[0], min_x_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(white_background, text, (min_x_tickmark[0]-20, min_x_tickmark[1]+30))
        elif i == 3:  # max X
            cv2.circle(white_background, (max_x_tickmark[0], max_x_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(white_background, text, (max_x_tickmark[0]-20, max_x_tickmark[1]+30))

    return white_background

def trace_original_image(cleaned_image, result_image, x_axis, y_axis, filtered_contours, 
                                  min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark,
                                  measures, original_with_traces,original_image):
    # Create a white image of the same size as the original

    # Define the bounding rectangle
    min_x, max_x = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    min_y, max_y = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])

    # Draw the axes in red
    cv2.line(original_with_traces, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
    cv2.line(original_with_traces, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 2)

    # Draw the filtered contours in green
    for curve in filtered_contours:
        curve_np = np.array(curve, dtype=np.int32)
        cv2.polylines(original_with_traces[min_y:max_y, min_x:max_x], [curve_np], False, (0, 255, 0), 2)

    # Function to add text with a white background
    def put_text_with_background(img, text, position, font_scale=0.9, thickness=2, text_color=(255,0,0), bg_color=(255,255,255)):
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_offset_x, text_offset_y = position
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    # Draw green dots and value labels for max and min locations
    for i, measure in enumerate(measures):
        text, _, _ = get_text_in_region(original_image, measure, i)
        if i == 0:  # max Y
            cv2.circle(original_with_traces, (max_y_tickmark[0], max_y_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(original_with_traces, text, (max_y_tickmark[0]-70, max_y_tickmark[1]+10))
        elif i == 1:  # min Y
            cv2.circle(original_with_traces, (min_y_tickmark[0], min_y_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(original_with_traces, text, (min_y_tickmark[0]-70, min_y_tickmark[1]+10))
        elif i == 2:  # min X
            cv2.circle(original_with_traces, (min_x_tickmark[0], min_x_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(original_with_traces, text, (min_x_tickmark[0]-20, min_x_tickmark[1]+30))
        elif i == 3:  # max X
            cv2.circle(original_with_traces, (max_x_tickmark[0], max_x_tickmark[1]), 5, (0, 255, 0), -1)
            put_text_with_background(original_with_traces, text, (max_x_tickmark[0]-20, max_x_tickmark[1]+30))

    return original_with_traces

def is_red_pixel(b, g, r):
    return r > 200 and g < 50 and b < 50

def save_debug_image(bitmap, filename):
    # Convert boolean or 0-1 float to 0-255 uint8
    if bitmap.dtype == bool or (bitmap.dtype in [np.float32, np.float64] and bitmap.max() <= 1):
        image_to_save = (1 - bitmap.astype(np.uint8)) * 255
    else:
        image_to_save = bitmap.astype(np.uint8)
    
    cv2.imwrite(filename, image_to_save)

def apply_secondary_filter(group, prev_groups):
    start_y, end_y, _ = group
    current_group_pixels = set(range(min(start_y, end_y), max(start_y, end_y) + 1))
    
    prev_groups_pixels = set()
    for prev_start_y, prev_end_y, _ in prev_groups:
        prev_groups_pixels.update(range(min(prev_start_y, prev_end_y), max(prev_start_y, prev_end_y) + 1))
    
    return bool(current_group_pixels.intersection(prev_groups_pixels))

def identify_pixel_groups(bitmap, column, min_height, start_points, end_points, prev_groups):
    height = bitmap.shape[0]
    groups = []
    final_proposed_groups = []
    in_group = False
    start_y = 0
    
    start_ys = set(y for x, y in start_points if x == column)
    end_ys = set(y for x, y in end_points if x == column)
    
    for y in range(height - 1, -1, -1):
        if bitmap[y, column] == 0 and not in_group:
            in_group = True
            start_y = y
        elif (bitmap[y, column] == 255 or y == 0) and in_group:
            end_y = y if bitmap[y, column] == 255 else y + 1
            group_height = start_y - end_y + 1
            
            contains_start_end = any(py in start_ys or py in end_ys for py in range(end_y, start_y + 1))
            
            if group_height >= min_height or contains_start_end:
                group = (start_y, end_y, group_height)
                groups.append(group)
                
                if contains_start_end:
                    final_proposed_groups.append(group)
                elif prev_groups:
                    would_keep = apply_secondary_filter(group, prev_groups)
                    if would_keep:
                        final_proposed_groups.append(group)
                    # else:
                    #     # Log when a group would be eliminated by the secondary filter
                    #     print(f"\nColumn {column}: Secondary filter eliminated a group")
                    #     print(f"prev_groups: {prev_groups}")
                    #     print(f"current_groups: {groups}")
                    #     print(f"final_proposed_groups: {final_proposed_groups}")
                else:
                    final_proposed_groups.append(group)
            else:
                for py in range(end_y, start_y + 1):
                    bitmap[py, column] = 255  # Set to white (eliminate)
            in_group = False
    
    # Update bitmap to remove eliminated groups
    eliminated_groups = [group for group in groups if group not in final_proposed_groups]
    for group in eliminated_groups:
        start_y, end_y, _ = group
        for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
            bitmap[y, column] = 255  # Set to white (eliminate)

    # Then return as before
    return len(final_proposed_groups), final_proposed_groups

def initialize_curve_thicknesses(bitmap, start_points, end_points):
    right_start = max(point[0] for point in start_points)
    left_end = min(point[0] for point in end_points)
    
    sample_range = left_end - right_start
    num_samples = 10
    sample_step = max(1, sample_range // num_samples)
    
    expected_groups = len(start_points)
    valid_thicknesses = []
    
    for x in range(right_start, left_end, sample_step):
        group_count, groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])  # Use empty list for prev_groups
        
        if group_count == expected_groups:
            avg_thickness = sum(group[2] for group in groups) / group_count
            valid_thicknesses.append(avg_thickness)
    
    if valid_thicknesses:
        return sum(valid_thicknesses) / len(valid_thicknesses)
    else:
        return None  # Or some default value