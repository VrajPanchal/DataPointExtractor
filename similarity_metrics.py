import numpy as np
import cv2
#from skimage.metrics import structural_similarity as ssim
#from skimage.feature import match_template
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

class AdvancedGraphSimilarity:
    """Advanced similarity metrics specifically designed for graph comparison"""
    
    def __init__(self):
        self.weight_structural = 0.4
        self.weight_visual = 0.3
        self.weight_data = 0.3
    
    def comprehensive_similarity(self, target_img: np.ndarray, generated_img: np.ndarray, 
                               target_data: Dict = None, generated_data: Dict = None) -> float:
        """Calculate comprehensive similarity score"""
        
        scores = {}
        
        # 1. Structural Similarity (SSIM)
        scores['ssim'] = self.structural_similarity(target_img, generated_img)
        
        # 2. Visual Feature Similarity
        scores['visual'] = self.visual_feature_similarity(target_img, generated_img)
        
        # 3. Data Pattern Similarity
        scores['data_pattern'] = self.data_pattern_similarity(target_img, generated_img)
        
        # 4. Color Distribution Similarity (if color images)
        if len(target_img.shape) == 3:
            scores['color'] = self.color_similarity(target_img, generated_img)
        else:
            scores['color'] = 1.0  # Perfect score for grayscale
        
        # Weighted combination
        final_score = (
            self.weight_structural * scores['ssim'] +
            self.weight_visual * scores['visual'] +
            self.weight_data * scores['data_pattern']
        )
        
        return final_score, scores
    
    def structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images"""
        # Ensure same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        return ssim(img1, img2)
    
    def visual_feature_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare visual features like edges, contours, and shapes"""
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1.copy()
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2.copy()
        
        # Resize if needed
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Edge detection
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        # Compare edge patterns
        edge_similarity = np.sum(edges1 & edges2) / np.sum(edges1 | edges2) if np.sum(edges1 | edges2) > 0 else 0
        
        # Contour analysis
        contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Compare number of contours (curves)
        contour_similarity = 1.0 - abs(len(contours1) - len(contours2)) / max(len(contours1), len(contours2), 1)
        
        return (edge_similarity + contour_similarity) / 2
    
    def data_pattern_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Analyze data patterns like curve shapes and marker positions"""
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1.copy()
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2.copy()
        
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Detect circular markers (data points)
        circles1 = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, 1, 20, 
                                   param1=50, param2=30, minRadius=2, maxRadius=15)
        circles2 = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=2, maxRadius=15)
        
        # Compare number of detected markers
        n1 = len(circles1[0]) if circles1 is not None else 0
        n2 = len(circles2[0]) if circles2 is not None else 0
        marker_similarity = 1.0 - abs(n1 - n2) / max(n1, n2, 1)
        
        # Analyze intensity distribution (curve density)
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Compare histograms
        hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return (marker_similarity + hist_similarity) / 2
    
    def color_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compare color distributions for colored graphs"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calculate color histograms for each channel
        similarity_scores = []
        
        for i in range(img1.shape[2]):
            hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            similarity_scores.append(similarity)
        
        return np.mean(similarity_scores)
    
    def extract_curve_features(self, img: np.ndarray) -> Dict:
        """Extract detailed features about curves in the graph"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        features = {}
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours (potential curves)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features['num_curves'] = len(contours)
        features['curve_lengths'] = [cv2.arcLength(contour, False) for contour in contours]
        features['curve_areas'] = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
        
        # Detect markers
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=1, maxRadius=10)
        features['num_markers'] = len(circles[0]) if circles is not None else 0
        
        # Analyze intensity patterns
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        return features
    
    def compare_extracted_features(self, features1: Dict, features2: Dict) -> float:
        """Compare extracted features between two graphs"""
        score = 0.0
        comparisons = 0
        
        # Compare number of curves
        if 'num_curves' in features1 and 'num_curves' in features2:
            max_curves = max(features1['num_curves'], features2['num_curves'], 1)
            curve_score = 1.0 - abs(features1['num_curves'] - features2['num_curves']) / max_curves
            score += curve_score
            comparisons += 1
        
        # Compare number of markers
        if 'num_markers' in features1 and 'num_markers' in features2:
            max_markers = max(features1['num_markers'], features2['num_markers'], 1)
            marker_score = 1.0 - abs(features1['num_markers'] - features2['num_markers']) / max_markers
            score += marker_score
            comparisons += 1
        
        # Compare edge density
        if 'edge_density' in features1 and 'edge_density' in features2:
            edge_diff = abs(features1['edge_density'] - features2['edge_density'])
            edge_score = max(0, 1.0 - edge_diff * 10)  # Scale factor for edge density
            score += edge_score
            comparisons += 1
        
        return score / comparisons if comparisons > 0 else 0.0

class GraphCodeAnalyzer:
    """Analyze and extract information from matplotlib code"""
    
    def extract_code_features(self, code: str) -> Dict:
        """Extract features from matplotlib code"""
        features = {}
        lines = code.split('\n')
        
        # Count plot commands
        plot_lines = [line for line in lines if 'ax.plot(' in line or 'plt.plot(' in line]
        features['num_plots'] = len(plot_lines)
        
        # Analyze colors used
        colors = []
        markers = []
        for line in plot_lines:
            # Simple extraction - in practice you'd want more robust parsing
            if "'" in line:
                style_part = line.split("'")[1]
                if len(style_part) > 0 and style_part[0] in 'rgbkmcyw':
                    colors.append(style_part[0])
                if len(style_part) > 1 and style_part[1] in 'o^v<>sd+x':
                    markers.append(style_part[1])
        
        features['colors_used'] = list(set(colors))
        features['markers_used'] = list(set(markers))
        features['num_unique_colors'] = len(set(colors))
        features['num_unique_markers'] = len(set(markers))
        
        # Check for axis labels and formatting
        features['has_xlabel'] = any('xlabel' in line for line in lines)
        features['has_ylabel'] = any('ylabel' in line for line in lines)
        features['has_grid'] = any('grid(' in line for line in lines)
        features['has_xlim'] = any('xlim(' in line for line in lines)
        features['has_ylim'] = any('ylim(' in line for line in lines)
        
        return features
    
    def compare_code_features(self, features1: Dict, features2: Dict) -> float:
        """Compare features extracted from two pieces of code"""
        score = 0.0
        comparisons = 0
        
        # Compare number of plots
        if 'num_plots' in features1 and 'num_plots' in features2:
            max_plots = max(features1['num_plots'], features2['num_plots'], 1)
            plot_score = 1.0 - abs(features1['num_plots'] - features2['num_plots']) / max_plots
            score += plot_score
            comparisons += 1
        
        # Compare colors
        if 'colors_used' in features1 and 'colors_used' in features2:
            colors1 = set(features1['colors_used'])
            colors2 = set(features2['colors_used'])
            intersection = len(colors1 & colors2)
            union = len(colors1 | colors2)
            color_score = intersection / union if union > 0 else 1.0
            score += color_score
            comparisons += 1
        
        # Compare formatting features
        format_features = ['has_xlabel', 'has_ylabel', 'has_grid', 'has_xlim', 'has_ylim']
        format_matches = sum(1 for feat in format_features 
                           if features1.get(feat, False) == features2.get(feat, False))
        format_score = format_matches / len(format_features)
        score += format_score
        comparisons += 1
        
        return score / comparisons if comparisons > 0 else 0.0

# Example usage
def demo_similarity_analysis():
    """Demonstrate the similarity analysis system"""
    
    # Create analyzer
    similarity_analyzer = AdvancedGraphSimilarity()
    code_analyzer = GraphCodeAnalyzer()
    
    # Example: analyze your target graph code
    target_code = '''
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], [18.5,16.2,15.8,12.1,10.8,9.2,7.8,6.1,4.2,2.8,1.2,0.5,0.2,0.1,0.05,0.02], 'ko-', markersize=6, linewidth=2)
ax.plot([1,2,3,4,5,6,7,8], [9.8,7.2,4.1,3.2,2.1,1.2,0.5,0.1], 'bo-', markersize=6, linewidth=2)
ax.plot([1,2,3,4,5,6], [7.2,4.8,2.1,1.8,0.8,0.2], 'go-', markersize=6, linewidth=2)
ax.plot([1,2,3,4], [3.8,1.2,0.5,0.1], 'ro-', markersize=6, linewidth=2)
ax.set_xlabel('Density (g cm⁻³)', fontsize=12)
ax.set_ylabel('Electronic energy gap (eV)', fontsize=12)
ax.set_xlim(1, 16)
ax.set_ylim(0, 20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''
    
    # Analyze target code features
    target_features = code_analyzer.extract_code_features(target_code)
    print("Target code features:")
    for key, value in target_features.items():
        print(f"  {key}: {value}")
    
    # Example generated code
    generated_code = '''
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [17.2,15.1,14.2,11.8,9.9,8.1,6.8,5.2,3.8,2.1,0.8,0.3,0.1,0.05,0.01], 'ko-', markersize=6, linewidth=2)
ax.plot([1,2,3,4,5,6,7], [8.9,6.8,3.8,2.9,1.8,0.9,0.3], 'bo-', markersize=6, linewidth=2)
ax.plot([1,2,3,4,5], [6.8,4.2,1.8,1.2,0.4], 'go-', markersize=6, linewidth=2)
ax.set_xlabel('Density (g cm⁻³)', fontsize=12)
ax.set_ylabel('Electronic energy gap (eV)', fontsize=12)
ax.set_xlim(1, 16)
ax.set_ylim(0, 20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''
    
    # Analyze generated code features
    generated_features = code_analyzer.extract_code_features(generated_code)
    print("\nGenerated code features:")
    for key, value in generated_features.items():
        print(f"  {key}: {value}")
    
    # Compare features
    feature_similarity = code_analyzer.compare_code_features(target_features, generated_features)
    print(f"\nCode feature similarity: {feature_similarity:.4f}")
    
    return target_code, generated_code

if __name__ == "__main__":
    demo_similarity_analysis()