import numpy as np
import cv2
from measure_detection import get_text_in_region

def trace_curves(cleaned_image, result_image, x_axis, y_axis):
    print("Starting trace_curves function")
    
    # Define the bounding rectangle
    min_x, max_x = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    min_y, max_y = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
    
    # Extract the ROI containing the curves from the cleaned image
    roi = cleaned_image[min_y:max_y, min_x:max_x]
    cv2.imwrite('debug_1_roi.png', roi)
    
    # Apply edge detection
    edges = cv2.Canny(roi, 150, 300)
    cv2.imwrite('debug_2_canny_edges.png', edges)
    
    # Apply dilation to merge double lines
    kernel_size = 2  # Adjust this value as needed
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite('debug_3_dilated_edges.png', dilated_edges)
    
    # Optional: Apply erosion to thin the lines if they became too thick
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    cv2.imwrite('debug_4_eroded_edges.png', eroded_edges)
    
    # Find starting points for each curve
    starting_points = find_curve_starting_points(eroded_edges, x_axis, y_axis)
    
    # Process each curve individually
    curves = []
    for start_point in starting_points:
        curve = trace_single_curve(eroded_edges, start_point)
        curves.append(curve)

    # Draw filtered contours on the result image
    for curve in curves:
        curve_np = np.array(curve, dtype=np.int32)
        cv2.polylines(result_image[min_y:max_y, min_x:max_x], [curve_np], False, (0, 255, 0), 1)
    
    cv2.imwrite('debug_7_result_with_contours.png', result_image)
    
    print("Finishing trace_curves function")
    return result_image, curves


def find_curve_starting_points(edges, x_axis, y_axis):
    starting_points = []
    height, width = edges.shape
    
    # Check along x-axis
    for x in range(width):
        if edges[0, x] > 0:  # Check top edge
            starting_points.append((x, 0))
        if edges[-1, x] > 0:  # Check bottom edge
            starting_points.append((x, height - 1))
    
    # Check along y-axis
    for y in range(height):
        if edges[y, 0] > 0:  # Check left edge
            starting_points.append((0, y))
        if edges[y, -1] > 0:  # Check right edge
            starting_points.append((width - 1, y))
    
    # Filter out noise and ensure minimum curve length
    min_curve_length = min(height, width) // 10  # Adjust as needed
    filtered_starting_points = []
    for point in starting_points:
        if is_valid_curve_start(edges, point, min_curve_length):
            filtered_starting_points.append(point)
    
    return filtered_starting_points

def is_valid_curve_start(edges, start, min_length):
    x, y = start
    length = 0
    visited = set()
    
    while 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0] and edges[y, x] > 0:
        length += 1
        visited.add((x, y))
        if length >= min_length:
            return True
        
        # Check neighboring pixels
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) not in visited and 0 <= new_x < edges.shape[1] and 0 <= new_y < edges.shape[0] and edges[new_y, new_x] > 0:
                x, y = new_x, new_y
                break
        else:
            break  # No unvisited neighbors found
    
    return False

def trace_single_curve(edges, start):
    curve = [start]
    x, y = start
    visited = set([start])
    
    while True:
        next_point = find_next_point(edges, (x, y), curve, visited)
        if next_point is None:
            break
        curve.append(next_point)
        visited.add(next_point)
        x, y = next_point
    
    return curve

def find_next_point(edges, current, curve, visited):
    x, y = current
    if len(curve) > 1:
        dx, dy = x - curve[-2][0], y - curve[-2][1]
        priority = [(dx, dy), (dy, -dx), (-dy, dx), (-dx, -dy)]
    else:
        priority = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    for dx, dy in priority:
        new_x, new_y = x + dx, y + dy
        if (new_x, new_y) not in visited and 0 <= new_x < edges.shape[1] and 0 <= new_y < edges.shape[0] and edges[new_y, new_x] > 0:
            return (new_x, new_y)
    return None

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

def trace_curve(start_point, bitmap, intersection_starts, intersection_ends, start_points,end_points, width):
    curve = [start_point]
    x, y = start_point

    # Get the initial group
    _, initial_groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
    current_group = next((group for group in initial_groups if y in range(min(group[0], group[1]), max(group[0], group[1]) + 1)), None)

    if current_group is None:
        print(f"Warning: No valid starting group found for point {start_point}")
        return curve

    while x < width - 1:
        x += 1
        # Check if we're entering an intersection
        intersection_index = next((i for i, start in enumerate(intersection_starts) if start == x), None)
        
        if intersection_index is not None:
            # We're entering an intersection, use prediction
            intersection_start = intersection_starts[intersection_index]
            intersection_end = intersection_ends[intersection_index]
            
            # Calculate gradient
            col_start = max(0, intersection_start - 11)
            col_end = intersection_start - 1
            _, groups_start = identify_pixel_groups(bitmap, col_start, 1, start_points, end_points, [])
            _, groups_end = identify_pixel_groups(bitmap, col_end, 1, start_points, end_points, [])
            
            curve_index = start_points.index(start_point)
            group_start = groups_start[curve_index % len(groups_start)]
            group_end = groups_end[curve_index % len(groups_end)]
            
            gradient_top = (group_end[0] - group_start[0]) / (col_end - col_start)
            gradient_bottom = (group_end[1] - group_start[1]) / (col_end - col_start)
            
            print(f"Curve starting at {start_point}:")
            print(f"  Intersection at x={intersection_start}")
            print(f"  Gradient calculation:")
            print(f"    Start column: {col_start}, End column: {col_end}")
            print(f"    Start group: {group_start}, End group: {group_end}")
            print(f"    Calculated gradients - Top: {gradient_top:.4f}, Bottom: {gradient_bottom:.4f}")
            
            # Predict through intersection and two columns beyond
            start_top_y, start_bottom_y = group_end[0], group_end[1]
            for col in range(intersection_start, intersection_end + 1):  
                steps = col - col_end
                predicted_top_y = round(start_top_y + gradient_top * steps)
                predicted_bottom_y = round(start_bottom_y + gradient_bottom * steps)
                middle_y = (predicted_top_y + predicted_bottom_y) // 2
                curve.append((col, middle_y))
            
            # Get the last predicted point
            last_predicted_x, last_predicted_y = curve[-1]
            
            # Find the group containing the last predicted point
            _, groups = identify_pixel_groups(bitmap, last_predicted_x, 1, start_points, end_points, [])
            current_group = next((group for group in groups if last_predicted_y in range(min(group[0], group[1]), max(group[0], group[1]) + 1)), None)
            
            if current_group is None:
                print(f"Warning: No valid group found for last predicted point ({last_predicted_x}, {last_predicted_y})")
                return curve
            
            # Move to the next column
            x = last_predicted_x + 1
            
            # Find connected group in the next column
            _, next_groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
            prev_y_set = set(range(min(current_group[0], current_group[1]), max(current_group[0], current_group[1]) + 1))
            connected_group = next((group for group in next_groups if prev_y_set.intersection(set(range(min(group[0], group[1]), max(group[0], group[1]) + 1)))), None)
            
            if connected_group is None:
                print(f"Warning: No connected group found after intersection at x={x}")
                return curve
            
            # Use the middle pixel of the connected group as the next trace point
            middle_y = (connected_group[0] + connected_group[1]) // 2
            curve.append((x, middle_y))
            current_group = connected_group
            
            continue  # Skip to the next iteration to start normal tracing from the new x

        # Normal tracing
        prev_y_set = set(range(min(current_group[0], current_group[1]), max(current_group[0], current_group[1]) + 1))
        _, current_groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
        
        connected_group = next((group for group in current_groups if prev_y_set.intersection(set(range(min(group[0], group[1]), max(group[0], group[1]) + 1)))), None)
        
        if connected_group is None:
            break
        
        middle_y = (connected_group[0] + connected_group[1]) // 2
        curve.append((x, middle_y))
        current_group = connected_group
        
        # Check if we've reached an end point
        if (x, middle_y) in end_points:
            break

    return curve

def scale_points(start_points, end_points, bitmap, intersection_starts, intersection_ends, x_scale, y_scale, origin, y_max, width):
    adjusted_points = []
    final_points = {}
    for j, start_point in enumerate(start_points):
        scaled_points = []
        traced_curve = trace_curve(start_point, bitmap, intersection_starts, intersection_ends, start_points, end_points, width)

        for point in traced_curve:

            adjusted_x = point[0] - origin[0]
            adjusted_y = origin[1] - point[1]    # this is flipped because y axis is positive down
            adjusted_points.append((adjusted_x, adjusted_y))

            scaled_x = adjusted_x * x_scale
            scaled_y = adjusted_y * y_scale + y_max
            if scaled_x >= 0 and scaled_y >= 0:
                scaled_points.append((scaled_x, scaled_y))

        final_points[j] = scaled_points

    return final_points

def find_axis_scales(min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark, measures,original_image):
    x_location_diff = max(max_x_tickmark[0], min_x_tickmark[0]) - min(max_x_tickmark[0], min_x_tickmark[0])
    y_location_diff = max(max_y_tickmark[1], min_y_tickmark[1]) - min(max_y_tickmark[1], min_y_tickmark[1])  

    text_measures = []
    for i, measure in enumerate(measures):
            text, _, _ = get_text_in_region(original_image, measure, i)
            text_measures.append(text)  

    max_y_measure, min_y_measure = float(text_measures[0]), float(text_measures[1])
    min_x_measure, max_x_measure = float(text_measures[2]), float(text_measures[3])

    y_measure_diff = max_y_measure - min_y_measure
    x_measure_diff = max_x_measure - min_x_measure

    x_scale = x_measure_diff / x_location_diff
    y_scale = y_measure_diff / y_location_diff

    origin = (min(max_x_tickmark[0], min_x_tickmark[0]), min(max_y_tickmark[1], min_y_tickmark[1]))

    return x_scale, y_scale, origin, max_y_measure