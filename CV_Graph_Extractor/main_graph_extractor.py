import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from image_preprocessing import remove_pixels
from axes_detection import detect_axes, extend_axes, find_tickmarks, remove_axes_and_ticks
from measure_detection import find_measures, get_text_in_region
from curve_tracing import trace_curves, trace_curve, scale_points, find_axis_scales
from contour_analysis import (
    create_debug_image, create_white_background_image, trace_original_image,
    is_red_pixel, save_debug_image, apply_secondary_filter, identify_pixel_groups, initialize_curve_thicknesses
)

if __name__ == "__main__":
    print("Script started")
    image_path = 'Chart_Image_1.jpg'
    print("Loading original image...")
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise FileNotFoundError(f"Image file not found.")
    original_with_traces = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    plt.imshow(original_image, cmap='gray')
    plt.show()
    print("Removing pixels in the 80-255 range...")
    processed_image = remove_pixels(original_image.copy())
    print("Pixels removed.")
    plt.imshow(processed_image, cmap='gray')
    plt.show()
    print("Detecting axes...")
    x_axis, y_axis = detect_axes(processed_image)
    print(f"Axes detected: x_axis={x_axis}, y_axis={y_axis}")
    print("Extending axes...")
    x_axis, y_axis = extend_axes(x_axis, y_axis)
    print(f"Axes extended: x_axis={x_axis}, y_axis={y_axis}")
    result_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    if x_axis is not None:
        cv2.line(result_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
    if y_axis is not None:
        cv2.line(result_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 2)
    print("Finding tickmarks...")
    min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark = find_tickmarks(processed_image, x_axis, y_axis)
    print(f"Tickmarks found: min_x_tickmark={min_x_tickmark}, max_x_tickmark={max_x_tickmark}, min_y_tickmark={min_y_tickmark}, max_y_tickmark={max_y_tickmark}")
    print("Finding measures...")
    measures = find_measures(original_image, x_axis, y_axis)
    print("Measures after find_measures:", measures)
    print("Number of measures found:", len(measures))
    for i, bbox in enumerate(measures):
        text, confidences, roi = get_text_in_region(original_image, bbox, i)
        print(f"Measure {i}: {bbox}")
        print(f"Text detected: '{text}'")
        print(f"Confidence scores: {confidences}")
        if not text:
            print(f"Warning: No text detected for measure {i}")
    cv2.imwrite('result_with_axes_and_measures.png', result_image)
    print("Intermediate result saved as 'result_with_axes_and_measures.png'")
    plt.imshow(result_image, cmap='gray')
    plt.show()
    print("Removing axes and ticks...")
    cleaned_image = remove_axes_and_ticks(processed_image, x_axis, y_axis)
    print("Axes and ticks removed.")
    print("About to call trace_curves")
    result_image, curves = trace_curves(cleaned_image, result_image, x_axis, y_axis)
    print("Finished calling trace_curves")
    roi_top_left = (min(x_axis[0], x_axis[2]), min(y_axis[1], y_axis[3]))
    roi_bottom_right = (max(x_axis[0], x_axis[2]), max(y_axis[1], y_axis[3]))
    print(f"ROI Top Left: ({int(roi_top_left[0])}, {int(roi_top_left[1])})")
    print(f"ROI Bottom Right: ({int(roi_bottom_right[0])}, {int(roi_bottom_right[1])})")
    highlighted_image = result_image.copy()
    print("Blue bounding box drawing skipped")
    min_x_tickmark = (int(min_x_tickmark[0]), int(min_x_tickmark[1]))
    max_x_tickmark = (int(max_x_tickmark[0]), int(max_x_tickmark[1]))
    min_y_tickmark = (int(min_y_tickmark[0]), int(min_y_tickmark[1]))
    max_y_tickmark = (int(max_y_tickmark[0]), int(max_y_tickmark[1]))
    print(f"min_x_tickmark: {min_x_tickmark}, max_x_tickmark: {max_x_tickmark}, min_y_tickmark: {min_y_tickmark}, max_y_tickmark: {max_y_tickmark}")
    cv2.circle(highlighted_image, min_x_tickmark, 10, (0, 255, 0), -1)
    cv2.circle(highlighted_image, max_x_tickmark, 10, (0, 255, 0), -1)
    cv2.circle(highlighted_image, min_y_tickmark, 10, (0, 255, 0), -1)
    cv2.circle(highlighted_image, max_y_tickmark, 10, (0, 255, 0), -1)
    print("Green dots for max and min locations drawn")
    for i, bbox in enumerate(measures):
        text, confidences, roi = get_text_in_region(result_image, bbox, i)
        if text:
            cv2.putText(highlighted_image, text, (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    print("Max and min values drawn in blue")
    if x_axis is not None:
        cv2.line(highlighted_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 5)
        print("Red x-axis drawn")
    if y_axis is not None:
        cv2.line(highlighted_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 0, 255), 5)
        print("Red y-axis drawn")
    cv2.imwrite('result_with_highlights.png', highlighted_image)
    print("Result with highlights saved as 'result_with_highlights.png'")
    plt.imshow(highlighted_image, cmap='gray')
    plt.show()
    white_background_image = create_white_background_image(cleaned_image, result_image, x_axis, y_axis, curves, 
                                                           min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark,
                                                           measures,original_image)
    original_with_traces = trace_original_image(cleaned_image, result_image, x_axis, y_axis, curves, 
                                                           min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark,
                                                           measures, original_with_traces,original_image)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'Chart_Image_1.jpg')
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        exit(1)
    height, width = img.shape[:2]
    print(f"Image dimensions: {width} x {height}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_size = 3
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    thresh_value = 40
    _, binary = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)
    min_length = 282
    min_area = 570
    line_thickness = 7
    plt.imshow(binary, cmap='gray')
    plt.show()
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    edge_margin = 10
    for cnt in contours:
        length = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if (length > min_length and area > min_area and x > edge_margin and y > edge_margin and x + w < width - edge_margin and y + h < height - edge_margin):
            filtered_contours.append(cnt)
    contour_img = np.ones_like(img) * 255
    cv2.drawContours(contour_img, filtered_contours, -1, (0, 0, 255), line_thickness)
    contour_img_copy = contour_img.copy()
    plt.imshow(contour_img, cmap='gray')
    plt.show()
    contour_copy_path = os.path.join(script_dir, 'initial_contour_image_copy.jpg')
    cv2.imwrite(contour_copy_path, contour_img_copy)
    print(f"Initial contour image copy saved to: {contour_copy_path}")
    start_points = []
    scan_width = int(width * 0.2)
    for x in range(scan_width):
        y = 0
        while y < height:
            if is_red_pixel(*map(int, contour_img[y, x])):
                top = bottom = y
                while bottom < height - 1 and is_red_pixel(*map(int, contour_img[bottom + 1, x])):
                    bottom += 1
                connected = False
                if x > 0:
                    for check_y in range(max(0, top - 2), min(height, bottom + 3)):
                        if is_red_pixel(*map(int, contour_img[check_y, x - 1])):
                            connected = True
                            break
                    if not connected:
                        middle_y = (top + bottom) // 2
                        if is_red_pixel(*map(int, contour_img[middle_y, x - 1])):
                            connected = True
                if not connected:
                    center_y = (top + bottom) // 2
                    start_points.append((x, center_y))
                y = bottom + 1
            else:
                y += 1
    end_points = []
    scan_width = int(width * 0.1)
    for x in range(width - 1, width - scan_width - 1, -1):
        y = 0
        while y < height:
            if is_red_pixel(*map(int, contour_img[y, x])):
                top = bottom = y
                while bottom < height - 1 and is_red_pixel(*map(int, contour_img[bottom + 1, x])):
                    bottom += 1
                connected = False
                if x < width - 1:
                    for check_y in range(max(0, top - 2), min(height, bottom + 3)):
                        if is_red_pixel(*map(int, contour_img[check_y, x + 1])):
                            connected = True
                            break
                    if not connected:
                        middle_y = (top + bottom) // 2
                        if is_red_pixel(*map(int, contour_img[middle_y, x + 1])):
                            connected = True
                if not connected:
                    center_y = (top + bottom) // 2
                    end_points.append((x, center_y))
                y = bottom + 1
            else:
                y += 1
    for point in start_points:
        cv2.circle(contour_img, point, 10, (0, 165, 255), -1)
    for point in end_points:
        cv2.circle(contour_img, point, 10, (255, 255, 0), -1)
    _, bitmap = cv2.threshold(cv2.cvtColor(contour_img_copy, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    plt.imshow(contour_img, cmap='gray')
    plt.show()
    #save_debug_image(bitmap, 'debug_initial_bitmap.jpg')
    initial_thickness = initialize_curve_thicknesses(bitmap, start_points, end_points)
    if initial_thickness is None:
        print("Failed to initialize curve thicknesses")
        exit(1)
    print(f"Initial average curve thickness: {initial_thickness:.2f}")
    left_x = min(point[0] for point in start_points)
    right_x = max(point[0] for point in end_points)
    prev_group_count = len(start_points)
    avg_heights = [initial_thickness] * len(start_points)
    intersection_starts = []
    intersection_ends = []
    in_intersection = False
    prev_groups = []
    for x in range(left_x, right_x + 1):
        min_avg = min(avg_heights)
        min_height = max(1, int(min_avg * 0.6))
        group_count, groups = identify_pixel_groups(bitmap, x, min_height, start_points, end_points, prev_groups)
        if group_count == len(start_points):
            for i, group in enumerate(groups):
                avg_heights[i] = (avg_heights[i] * 0.1 + group[2] * 0.9)
        if not in_intersection and group_count < prev_group_count and prev_group_count > 0:
            if groups:
                max_height = max(group[2] for group in groups)
                avg_height = sum(avg_heights) / len(avg_heights) if avg_heights else initial_thickness
                if max_height > 1.8 * avg_height:
                    intersection_starts.append(x)
                    in_intersection = True
                    print(f"Intersection start detected at column {x}")
                    #save_debug_image(bitmap, f'debug_artifact_detected_column_{x}.jpg')
        elif in_intersection and group_count > 1:
            max_height = max(group[2] for group in groups)
            if max_height < 1.5 * initial_thickness:
                intersection_ends.append(x)
                in_intersection = False
                print(f"Intersection end detected at column {x}")
                #save_debug_image(bitmap, f'debug_artifact_cleaned_column_{x}.jpg')
        prev_group_count = group_count
        prev_groups = groups
    right_x = max(point[0] for point in end_points)
    left_x = min(point[0] for point in start_points)
    avg_heights_backward = avg_heights.copy()
    prev_groups = []
    print("\n\nEnding points:")
    for x, y in end_points:
        print(f"  ({x}, {y})")
    print(f"\n*** Starting reverse pass from x={right_x} to x={left_x} ***")
    print(f"Initial avg_heights_backward: {avg_heights_backward}")
    end_y_coords = set(y for _, y in end_points)
    for i, x in enumerate(range(right_x, max(left_x, right_x - 1000), -1)):
        group_count, groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, prev_groups)
        end_connected_groups = []
        other_groups = []
        for group in groups:
            if x in [point[0] for point in end_points]:
                start_y, end_y, _ = group
                group_y_set = set(range(min(start_y, end_y), max(start_y, end_y) + 1))
                if group_y_set.intersection(end_y_coords):
                    end_connected_groups.append(group)
                    continue
            other_groups.append(group)
        filtered_groups = end_connected_groups.copy()
        for group in other_groups:
            if apply_secondary_filter(group, prev_groups):
                filtered_groups.append(group)
            else:
                start_y, end_y, _ = group
                for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                    bitmap[y, x] = 255
                print(f"\nColumn {x}: Secondary filter removed a group {group}")
        eliminated_groups = [group for group in groups if group not in filtered_groups]
        for group in eliminated_groups:
            start_y, end_y, _ = group
            for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                bitmap[y, x] = 255
        in_intersection = any(right <= x <= left for left, right in zip(intersection_starts, intersection_ends))
        if len(filtered_groups) == len(end_points) and not in_intersection:
            for j, group in enumerate(filtered_groups):
                avg_heights_backward[j] = (avg_heights_backward[j] * 0.1 + group[2] * 0.9)
        prev_groups = filtered_groups
        if len(filtered_groups) > len(end_points):
            print(f'\n*** Too Many Groups in Column {x}')
            print(f'Groups: {filtered_groups}')
            break
    print("\nReverse pass completed")
    print(f"Final avg_heights_backward: {avg_heights_backward}")
    contour_img_updated = np.ones_like(img) * 255
    cv2.drawContours(contour_img_updated, filtered_contours, -1, (0, 0, 255), line_thickness)
    for point in start_points:
        cv2.circle(contour_img_updated, point, 10, (0, 165, 255), -1)
    for point in end_points:
        cv2.circle(contour_img_updated, point, 10, (255, 255, 0), -1)
    image_height = contour_img_updated.shape[0]
    for left, right in zip(intersection_starts, intersection_ends):
        cv2.line(contour_img_updated, (left, 0), (left, image_height), (0, 0, 0), 2)
        cv2.line(contour_img_updated, (right, 0), (right, image_height), (255, 0, 255), 2)
    #save_debug_image(bitmap, 'debug_final_bitmap_after_reverse_pass.jpg')
    plt.imshow(bitmap, cmap='gray')
    plt.show()
    is_anomaly = False
    anomaly_start_columns = []
    anomaly_end_columns = []
    anomaly_start_groups = []
    anomaly_end_groups = []
    running_avg_heights = avg_heights_backward.copy()
    left_x = min(point[0] for point in start_points)
    right_x = max(point[0] for point in end_points)
    print("\n*** Starting anomaly detection pass ***")
    print(f"Initial running_avg_heights: {running_avg_heights}")
    print(f"Start points: {start_points}")
    start_point_coords = set(start_points)
    prev_groups = []
    for x in range(left_x, right_x + 1):
        if not any(start <= x <= end for start, end in zip(intersection_starts, intersection_ends)):
            group_count, groups = identify_pixel_groups(bitmap, x, 1, start_points, end_points, [])
            if group_count == len(start_points):
                all_within_range = True
                for i, group in enumerate(groups):
                    start_y, end_y, height = group
                    is_start_point = (x, start_y) in start_point_coords or (x, end_y) in start_point_coords
                    if not is_anomaly and height > running_avg_heights[i] * 1.8 and not is_start_point:
                        is_anomaly = True
                        anomaly_start_columns.append(x)
                        anomaly_start_groups.append((prev_groups, groups))
                        print(f"  Anomaly start detected! Group height: {height}, Running avg: {running_avg_heights[i]}")
                    if abs(height - running_avg_heights[i]) > 0.2 * running_avg_heights[i]:
                        all_within_range = False
                if is_anomaly and all_within_range:
                    anomaly_end_columns.append(x)
                    anomaly_end_groups.append((prev_groups, groups))
                    print(f"  Anomaly end detected at column {x}")
                    is_anomaly = False
            if not is_anomaly:
                for i, group in enumerate(groups):
                    start_y, end_y, height = group
                    is_start_point = (x, start_y) in start_point_coords or (x, end_y) in start_point_coords
                    if not is_start_point:
                        old_avg = running_avg_heights[i]
                        running_avg_heights[i] = running_avg_heights[i] * 0.1 + height * 0.9
            prev_groups = groups
    print("\n*** Anomaly detection pass completed ***")
    if anomaly_start_columns:
        for i, (start, end) in enumerate(zip(anomaly_start_columns, anomaly_end_columns)):
            print(f"\nAnomaly {i+1}:")
            print(f"Start column: {start}")
            print(f"Previous groups at start: {anomaly_start_groups[i][0]}")
            print(f"Current groups at start: {anomaly_start_groups[i][1]}")
            print(f"End column: {end}")
            print(f"Previous groups at end: {anomaly_end_groups[i][0]}")
            print(f"Current groups at end: {anomaly_end_groups[i][1]}")
    else:
        print("No anomalies detected")
    for start in anomaly_start_columns:
        cv2.line(contour_img_updated, (start, 0), (start, image_height), (255, 0, 0), 2)
    for end in anomaly_end_columns:
        cv2.line(contour_img_updated, (end, 0), (end, image_height), (0, 255, 0), 2)
    plt.imshow(contour_img_updated, cmap='gray')
    plt.show()
    print("\n*** Starting smoothing process ***")
    if anomaly_start_columns:
        for region_index, (start_column, end_column) in enumerate(zip(anomaly_start_columns, anomaly_end_columns)):
            col_start = start_column - 21
            col_end = start_column - 1
            _, groups_start = identify_pixel_groups(bitmap, col_start, 1, start_points, end_points, [])
            _, groups_end = identify_pixel_groups(bitmap, col_end, 1, start_points, end_points, [])
            anomalous_group_index = next(i for i, (start, end) in enumerate(zip(anomaly_start_groups[region_index][0], anomaly_start_groups[region_index][1])) if end[2] > start[2] * 1.8)
            group_start = groups_start[anomalous_group_index]
            group_end = groups_end[anomalous_group_index]
            gradient_top = (group_end[0] - group_start[0]) / 20
            gradient_bottom = (group_end[1] - group_start[1]) / 20
            start_top_y, start_bottom_y, _ = group_end
            print(f"\nSmoothing Anomalous Region {region_index + 1}: columns {start_column} to {end_column}")
            print(f"Anomalous group index: {anomalous_group_index}")
            print(f"Start y-range: {start_top_y} to {start_bottom_y}")
            print(f"Gradient (top, bottom): {gradient_top}, {gradient_bottom}")
            print("\nProcessing columns:")
            for col in range(start_column, end_column):
                steps = col - col_end
                predicted_top_y = round(start_top_y + gradient_top * steps)
                predicted_bottom_y = round(start_bottom_y + gradient_bottom * steps)
                predicted_height = abs(predicted_bottom_y - predicted_top_y) + 1
                _, current_groups = identify_pixel_groups(bitmap, col, 1, start_points, end_points, [])
                anomalous_group = current_groups[anomalous_group_index]
                actual_top_y, actual_bottom_y, actual_height = anomalous_group
                predicted_min_y, predicted_max_y = min(predicted_top_y, predicted_bottom_y), max(predicted_top_y, predicted_bottom_y)
                actual_min_y, actual_max_y = min(actual_top_y, actual_bottom_y), max(actual_top_y, actual_bottom_y)
                pixels_changed = 0
                for y in range(actual_min_y, actual_max_y + 1):
                    if y < predicted_min_y or y > predicted_max_y:
                        if bitmap[y, col] == 0:
                            bitmap[y, col] = 255
                            pixels_changed += 1
                print(f"\nColumn {col}:")
                print(f"  Predicted group: ({predicted_min_y}, {predicted_max_y}, {predicted_height})")
                print(f"  Anomalous group: ({actual_min_y}, {actual_max_y}, {actual_height})")
                print(f"  Pixels changed: {pixels_changed}")
            print(f"\nSmoothing process completed for Region {region_index + 1}.")
        print("\nAll anomalous regions processed.")
        final_bitmap_path = os.path.join(script_dir, 'final_smoothed_bitmap.jpg')
        cv2.imwrite(final_bitmap_path, bitmap)
        print(f"Final smoothed bitmap saved to: {final_bitmap_path}")
    else:
        print("No anomalies detected, no smoothing necessary.")
    plt.imshow(bitmap, cmap='gray')
    plt.show()
    final_output_path = os.path.join(script_dir, 'final_result_with_anomalies.jpg')
    cv2.imwrite(final_output_path, contour_img_updated)
    print(f"Final image with anomaly detections saved to: {final_output_path}")
    print("\n*** Starting curve tracing process ***")
    additional_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for j, start_point in enumerate(start_points):
        traced_curve = trace_curve(start_point, bitmap, intersection_starts, intersection_ends,start_points,end_points,width)
        final_image = cv2.imread(final_output_path)
        if final_image is not None:
            for i in range(1, len(traced_curve)):
                cv2.line(white_background_image, traced_curve[i-1], traced_curve[i], additional_colors[j % len(additional_colors)], 3)
            for i in range(1, len(traced_curve)):
                cv2.line(original_with_traces, traced_curve[i-1], traced_curve[i], additional_colors[j % len(additional_colors)], 3)
        else:
            print(f"Error: Unable to load the final image from {final_output_path}")
            break
    plt.imshow(original_with_traces)
    plt.show()
    plt.imshow(white_background_image)
    plt.show()
    cv2.imwrite('white_background_traced_elements.png', white_background_image)
    print("White background image with traced elements saved as 'white_background_traced_elements.png'")
    x_scale, y_scale, origin, y_max = find_axis_scales(min_x_tickmark, max_x_tickmark, min_y_tickmark, max_y_tickmark, measures,original_image)
    final_points = scale_points(start_points, end_points, bitmap, intersection_starts, intersection_ends, x_scale, y_scale, origin, y_max, width)
    df = pd.DataFrame()
    for curve_number, coordinates in final_points.items():
        curve_df = pd.DataFrame(coordinates, columns=[f'x_curve_{curve_number}', f'y_curve_{curve_number}'])
        df = pd.concat([df, curve_df], axis=1)
    df = df.dropna()
    df.to_excel('output.xlsx', index=False)
    print(f'X-scale: {x_scale} per pixel')
    print(f'Y-scale: {y_scale} per pixel')
    print(f'Origin: {origin}') 