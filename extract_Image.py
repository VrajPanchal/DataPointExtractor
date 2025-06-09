import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import io
from PIL import Image
import argparse
from pathlib import Path

def pdf_to_images(pdf_path, output_dir, dpi=200):
    """
    Convert PDF pages to images
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save images
        dpi (int): Resolution for image conversion
    
    Returns:
        list: List of image file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open PDF
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    
    print(f"Converting PDF to images...")
    
    for page_num in range(len(pdf_document)):
        # Get page
        page = pdf_document.load_page(page_num)
        
        # Create transformation matrix for desired DPI
        mat = fitz.Matrix(dpi/72, dpi/72)
        
        # Render page to image
        pix = page.get_pixmap(matrix=mat)
        
        # Save image
        image_filename = f"page_{page_num + 1:03d}.png"
        image_path = os.path.join(output_dir, image_filename)
        pix.save(image_path)
        image_paths.append(image_path)
        print(f"Saved: {image_filename}")
    
    pdf_document.close()
    return image_paths

def detect_graphs_plots(image_path, output_dir, detection_dpi=35, dpi=200):
    """
    Detect and extract graphs/plots from an image using computer vision
    Two-pass approach: detect at low DPI, extract at high DPI
    
    Args:
        image_path (str): Path to the high-resolution image file
        output_dir (str): Directory to save extracted graphs
        detection_dpi (int): DPI to use for detection (lower for better accuracy)
    
    Returns:
        list: List of extracted graph image paths
    """
    # Read high-resolution image
    img_high = cv2.imread(image_path)
    if img_high is None:
        return []
    
    # Create low-resolution version for detection
    scale_factor = detection_dpi / 200  # Assuming original is 200 DPI
    new_width = int(img_high.shape[1] * scale_factor)
    new_height = int(img_high.shape[0] * scale_factor)
    img_low = cv2.resize(img_high, (new_width, new_height), interpolation=cv2.INTER_AREA)
    gray_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY)
    
    extracted_graphs = []
    
    # Method 1: Detect rectangular regions (common for plots/graphs)
    # Apply edge detection on low-res image
    edges = cv2.Canny(gray_low, 30, 100, apertureSize=3)
    
    # Dilate edges to connect nearby lines
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio (typical for graphs)
    min_area = (img_low.shape[0] * img_low.shape[1]) * 0.02  # At least 2% of image
    max_area = (img_low.shape[0] * img_low.shape[1]) * 0.7   # At most 70% of image
    
    graph_candidates = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (graphs are usually wider than tall)
            aspect_ratio = w / h
            
            # Filter by size and aspect ratio - more restrictive for low DPI
            if w > 20 and h > 20 and 0.3 < aspect_ratio < 4:
                # Check if region contains graph-like features
                if is_likely_graph(gray_low[y:y+h, x:x+w]):
                    graph_candidates.append((x, y, w, h, area))
    
    # Sort by area (largest first) and take top candidates
    graph_candidates.sort(key=lambda x: x[4], reverse=True)
    
    # Method 2: Look for grid patterns (common in graphs)
    # Detect horizontal and vertical lines with smaller kernels for low DPI
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    
    horizontal_lines = cv2.morphologyEx(gray_low, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(gray_low, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine horizontal and vertical lines
    grid_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
    
    # Find regions with grid patterns
    grid_contours, _ = cv2.findContours(grid_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in grid_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:
                # Check if it's actually a graph, not just text with lines
                region = gray_low[y:y+h, x:x+w]
                if is_likely_graph(region):
                    graph_candidates.append((x, y, w, h, area))
    
    # Remove duplicates and overlapping regions
    final_candidates = []
    for candidate in graph_candidates:
        x, y, w, h, area = candidate
        is_duplicate = False
        
        for existing in final_candidates:
            ex, ey, ew, eh, _ = existing
            
            # Check for significant overlap
            overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
            overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
            overlap_area = overlap_x * overlap_y
            
            if overlap_area > 0.3 * min(w * h, ew * eh):
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_candidates.append(candidate)
    
    # Convert coordinates back to high-resolution image
    scale_up = 1 / scale_factor
    final_candidates_high_res = []
    
    for x, y, w, h, area in final_candidates:
        x_high = int(x * scale_up)
        y_high = int(y * scale_up)
        w_high = int(w * scale_up)
        h_high = int(h * scale_up)
        final_candidates_high_res.append((x_high, y_high, w_high, h_high, area))
    
    # Extract and save the detected graphs from high-resolution image
    page_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for i, (x, y, w, h, _) in enumerate(final_candidates_high_res[:5]):  # Limit to top 5
        # Add some padding (scaled for high-res)
        padding = int(20 * scale_up)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_high.shape[1] - x, w + 2 * padding)
        h = min(img_high.shape[0] - y, h + 2 * padding)
        
        # Extract the region from high-resolution image
        extracted_region = img_high[y:y+h, x:x+w]
        
        # Final check: ensure extracted region is likely a graph
        gray_region = cv2.cvtColor(extracted_region, cv2.COLOR_BGR2GRAY)
        if is_likely_graph(gray_region):
            # Save extracted graph
            graph_filename = f"{page_name}_graph_{i+1}.png"
            graph_path = os.path.join(output_dir, graph_filename)
            cv2.imwrite(graph_path, extracted_region)
            
            extracted_graphs.append(graph_path)
            print(f"Extracted graph: {graph_filename}")
    
    return extracted_graphs

def is_likely_graph(gray_region):
    """
    Determine if a region is likely to contain a graph rather than text
    
    Args:
        gray_region: Grayscale image region to analyze
    
    Returns:
        bool: True if likely a graph, False if likely text
    """
    if gray_region.size == 0:
        return False
        
    h, w = gray_region.shape
    
    # Skip very small regions
    if w < 50 or h < 50:
        return False
    
    # Check for line patterns typical of graphs
    # Look for horizontal lines (axes, grid lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w//10, 5), 1))
    horizontal_lines = cv2.morphologyEx(gray_region, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal_score = np.sum(horizontal_lines > 0) / gray_region.size
    
    # Look for vertical lines (axes, grid lines)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h//10, 5)))
    vertical_lines = cv2.morphologyEx(gray_region, cv2.MORPH_OPEN, vertical_kernel)
    vertical_score = np.sum(vertical_lines > 0) / gray_region.size
    
    # Check for text-like patterns (many small connected components)
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # Count small components (typical of text characters)
    small_components = 0
    medium_components = 0
    
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Small components typical of text
        if 10 < area < 500 and width < w//5 and height < h//5:
            small_components += 1
        # Medium components could be graph elements
        elif 500 < area < (w * h * 0.1):
            medium_components += 1
    
    # Calculate density of small components (high for text)
    small_component_density = small_components / (w * h / 1000)
    
    # Decision logic
    has_lines = horizontal_score > 0.001 or vertical_score > 0.001
    is_not_dense_text = small_component_density < 2.0
    has_graph_elements = medium_components >= 1
    
    # Additional check: look for continuous lines (graph axes/curves)
    edges = cv2.Canny(gray_region, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    long_lines = 0
    for contour in contours:
        if cv2.arcLength(contour, False) > min(w, h) * 0.3:  # Long relative to image size
            long_lines += 1
    
    has_long_lines = long_lines >= 1
    
    # Final decision: likely a graph if it has graph-like features and isn't dense text
    return (has_lines or has_long_lines or has_graph_elements) and is_not_dense_text

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to images and extract graphs/plots')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output-dir', '-o', default='output', help='Output directory (default: output)')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for image conversion (default: 200)')
    parser.add_argument('--format', choices=['png', 'jpeg'], default='png', help='Image format (default: png)')
    parser.add_argument('--extract-graphs', action='store_true', default=True, help='Extract graphs from images')
    
    args = parser.parse_args()
    
    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        return
    
    # Create output directories
    images_dir = os.path.join(args.output_dir, 'pages')
    graphs_dir = os.path.join(args.output_dir, 'extracted_graphs')
    
    print(f"Processing: {args.pdf_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Convert PDF to images
    image_paths = pdf_to_images(args.pdf_path, images_dir, args.dpi)
    
    print(f"\nConverted {len(image_paths)} pages to images.")
    
    # Extract graphs if requested
    if args.extract_graphs:
        print("\nExtracting graphs and plots...")
        os.makedirs(graphs_dir, exist_ok=True)
        
        total_graphs = 0
        for image_path in image_paths:
            extracted = detect_graphs_plots(image_path, graphs_dir, detection_dpi=35, dpi=args.dpi)
            total_graphs += len(extracted)
        
        print(f"\nExtracted {total_graphs} potential graphs/plots.")
        print(f"Images saved in: {images_dir}")
        print(f"Extracted graphs saved in: {graphs_dir}")
    else:
        print(f"Images saved in: {images_dir}")

if __name__ == "__main__":
    main()

# Example usage if running as a script:
# python extract_Image.py document.pdf --output-dir my_output --dpi 300 --format png