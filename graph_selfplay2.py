import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
# import torch # Not used in this version, can be removed if no PyTorch model is directly integrated
# import torch.nn as nn # Not used
# import torch.optim as optim # Not used
import random
from typing import List, Tuple, Dict
import json
import os
import math # For log checks

class GraphCodeGenerator:
    """Generates Python code for matplotlib plots"""

    def __init__(self):
        self.color_options = ['r', 'b', 'g', 'k', 'c', 'm', 'y']
        self.marker_options = ['o', 's', '^', 'v', '<', '>', 'D', 'x', '+', None] # Added None for no marker
        self.line_styles = ['-', '--', '-.', ':', None] # Added None for scatter-like plots
        self.scale_options = ['linear', 'log']
        self.font_size_options = [8, 10, 12, 14]
        self.grid_options = [True, False]

    def generate_random_code(self) -> str:
        """Generate random matplotlib code with more diverse options"""
        num_curves = random.randint(1, 4) # Reduced max for simplicity with log scales

        # Determine scales first, as they affect data generation and limits
        x_scale = random.choice(self.scale_options)
        y_scale = random.choice(self.scale_options)

        # Adjust data generation and limits for log scales
        x_min, x_max_gen = (0.1, 10) if x_scale == 'log' else (1, 16)
        y_min_gen, y_max_gen = (0.1, 100) if y_scale == 'log' else (0, 20) # Adjusted y_max for log

        # Ensure x_lim and y_lim are appropriate for the scale
        xlim_min = 0.01 if x_scale == 'log' else 0
        ylim_min = 0.01 if y_scale == 'log' else -5 # Allow slight negative for linear if data goes low

        xlim_max = x_max_gen * (1.5 if x_scale == 'log' else 1.1)
        ylim_max = y_max_gen * (1.5 if y_scale == 'log' else 1.2)


        code_parts = [
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "fig, ax = plt.subplots(figsize=(8, 6))"
        ]

        for i in range(num_curves):
            num_points = random.randint(5, 15)
            # Generate x points: ensure positive for log scale
            if x_scale == 'log':
                # Generate points in log space then exponentiate, or ensure positivity
                x_points = sorted([round(random.uniform(x_min, x_max_gen), 2) for _ in range(num_points)])
            else:
                x_points = sorted([round(random.uniform(x_min, x_max_gen), 2) for _ in range(num_points)])

            # Generate y points: ensure positive for log scale
            if y_scale == 'log':
                y_points = [round(random.uniform(y_min_gen, y_max_gen), 2) for _ in range(len(x_points))]
            else:
                y_points = [round(random.uniform(y_min_gen, y_max_gen), 2) for _ in range(len(x_points))]


            color = random.choice(self.color_options)
            marker = random.choice(self.marker_options)
            linestyle = random.choice(self.line_styles)
            if marker is None and linestyle is None: # Ensure it's either a line or has markers
                linestyle = '-'

            x_str = str(x_points)
            y_str = str(y_points)

            plot_line_parts = [f"ax.plot({x_str}, {y_str}"]
            plot_line_parts.append(f"color='{color}'")
            if marker:
                plot_line_parts.append(f"marker='{marker}'")
            if linestyle:
                plot_line_parts.append(f"linestyle='{linestyle}'")
            plot_line_parts.append("markersize=6")
            plot_line_parts.append("linewidth=2)")
            code_parts.append(", ".join(plot_line_parts))

        # Add formatting
        xlabel = random.choice(["Density (g cm^-3)", "Time (s)", "Concentration (mol/L)", "X-Value"])
        ylabel = random.choice(["Electrical conductivity (Ω⁻¹ m⁻¹)", "Density of States (eV⁻¹ atom⁻¹)", "Energy (eV)", "Y-Value"])
        title = random.choice(["Experimental Data", "Simulation Results", "Trend Analysis", ""])

        fontsize = random.choice(self.font_size_options)

        if title:
             code_parts.append(f"ax.set_title('{title}', fontsize={fontsize + 2})")
        code_parts.append(f"ax.set_xlabel('{xlabel}', fontsize={fontsize})")
        code_parts.append(f"ax.set_ylabel('{ylabel}', fontsize={fontsize})")

        if x_scale == 'log':
            code_parts.append("ax.set_xscale('log')")
        if y_scale == 'log':
            code_parts.append("ax.set_yscale('log')")

        # Ensure limits are valid for log scale (must be > 0)
        final_xlim_min = max(xlim_min, 1e-2) if x_scale == 'log' else xlim_min
        final_ylim_min = max(ylim_min, 1e-2) if y_scale == 'log' else ylim_min

        code_parts.append(f"ax.set_xlim({final_xlim_min}, {xlim_max})")
        code_parts.append(f"ax.set_ylim({final_ylim_min}, {ylim_max})")

        if random.choice(self.grid_options):
            code_parts.append("ax.grid(True, alpha=0.4, which='both', axis='both', linestyle=':')") # More specific grid

        if num_curves > 1 and random.random() < 0.7: # Add legend sometimes
            code_parts.append("ax.legend()")

        code_parts.append("plt.tight_layout()")
        return "\n".join(code_parts)

    def mutate_code(self, code: str) -> str:
        lines = code.split('\n')
        plot_lines_indices = [i for i, line in enumerate(lines) if 'ax.plot(' in line]

        if not plot_lines_indices:
            return self.generate_random_code()

        mutation_type = random.choice(['modify_data', 'change_style', 'add_curve', 'remove_curve', 'change_scale_limit_label', 'toggle_legend_grid'])

        if mutation_type == 'modify_data' and plot_lines_indices:
            line_idx = random.choice(plot_lines_indices)
            lines[line_idx] = self._modify_plot_data(lines[line_idx], lines)
        elif mutation_type == 'change_style' and plot_lines_indices:
            line_idx = random.choice(plot_lines_indices)
            lines[line_idx] = self._change_plot_style(lines[line_idx])
        elif mutation_type == 'add_curve':
            # Find where to insert: before labels/limits/grid/legend
            insert_idx = len(lines) - 1
            for i, l in reversed(list(enumerate(lines))):
                if any(s in l for s in ["ax.set_xlabel", "ax.set_title", "ax.set_xlim", "ax.set_xscale", "ax.grid", "ax.legend", "plt.tight_layout"]):
                    insert_idx = i
                else:
                    break
            lines.insert(insert_idx, self._generate_single_curve(lines))
        elif mutation_type == 'remove_curve' and len(plot_lines_indices) > 1:
            line_idx_to_remove = random.choice(plot_lines_indices)
            lines.pop(line_idx_to_remove)
        elif mutation_type == 'change_scale_limit_label':
            # Mutate scale
            if random.random() < 0.3: # Change scale
                if "ax.set_xscale('log')" in lines:
                    if random.random() < 0.5: lines.remove("ax.set_xscale('log')")
                elif "ax.set_xscale('linear')" in lines: # Should not be explicit, but handle
                     if random.random() < 0.5: lines[lines.index("ax.set_xscale('linear')")] = "ax.set_xscale('log')"
                else: # Is linear, try to make log
                    if random.random() < 0.5:
                        idx = self._find_line_index_before(lines, ["ax.set_xlim", "ax.set_xlabel"])
                        lines.insert(idx, "ax.set_xscale('log')")
                # Similar for y_scale
                if "ax.set_yscale('log')" in lines:
                    if random.random() < 0.5: lines.remove("ax.set_yscale('log')")
                elif "ax.set_yscale('linear')" in lines:
                     if random.random() < 0.5: lines[lines.index("ax.set_yscale('linear')")] = "ax.set_yscale('log')"
                else:
                    if random.random() < 0.5:
                        idx = self._find_line_index_before(lines, ["ax.set_ylim", "ax.set_ylabel"])
                        lines.insert(idx, "ax.set_yscale('log')")

            # Mutate limits (simple adjustment)
            for i, line in enumerate(lines):
                if "ax.set_xlim(" in line or "ax.set_ylim(" in line:
                    parts = line.split(',')
                    try:
                        val1 = float(parts[0].split('(')[1])
                        val2 = float(parts[1].split(')')[0])
                        adj1 = val1 * random.uniform(0.8, 1.2)
                        adj2 = val2 * random.uniform(0.8, 1.2)
                        # Ensure min < max and positive for log if applicable
                        current_scale = 'log' if ('xscale(\'log\')' in "\n".join(lines) and "xlim" in line) or \
                                              ('yscale(\'log\')' in "\n".join(lines) and "ylim" in line) else 'linear'
                        if current_scale == 'log':
                            adj1 = max(1e-2, adj1)
                            adj2 = max(1e-2, adj2)

                        if adj1 < adj2:
                             lines[i] = f"{parts[0].split('(')[0]}({adj1:.2f}, {adj2:.2f})" + ",".join(parts[1:]).split(')')[1] + ")"
                        else: # Swap if order got messed up
                             lines[i] = f"{parts[0].split('(')[0]}({adj2:.2f}, {adj1:.2f})" + ",".join(parts[1:]).split(')')[1] + ")"

                    except: pass # If parsing fails, skip
            # Mutate labels
            for i, line in enumerate(lines):
                if "ax.set_xlabel(" in line:
                    lines[i] = f"ax.set_xlabel('{random.choice(['Density', 'Time', 'X', 'Input'])}', fontsize={random.choice(self.font_size_options)})"
                elif "ax.set_ylabel(" in line:
                    lines[i] = f"ax.set_ylabel('{random.choice(['Conductivity', 'Output', 'Y', 'Value'])}', fontsize={random.choice(self.font_size_options)})"
        elif mutation_type == 'toggle_legend_grid':
            has_legend = any("ax.legend()" in l for l in lines)
            if has_legend:
                if random.random() < 0.5: lines = [l for l in lines if "ax.legend()" not in l] # remove
            else:
                if random.random() < 0.5: # add
                    idx = self._find_line_index_before(lines, ["plt.tight_layout"])
                    lines.insert(idx, "ax.legend()")
            has_grid = any("ax.grid(" in l for l in lines)
            if has_grid:
                if random.random() < 0.5: lines = [l for l in lines if "ax.grid(" not in l] # remove
            else:
                if random.random() < 0.5: # add
                    idx = self._find_line_index_before(lines, ["plt.tight_layout", "ax.legend"])
                    lines.insert(idx, f"ax.grid({random.choice(self.grid_options)}, alpha=0.4, which='both', axis='both', linestyle=':')")


        # Ensure scales are declared before limits/data if log scale is added
        new_code = "\n".join(lines)
        return self._reorder_plot_settings(new_code)

    def _find_line_index_before(self, lines: List[str], keywords: List[str]) -> int:
        """Helper to find an insertion point before certain keywords."""
        for i in range(len(lines) -1, -1, -1):
            for kw in keywords:
                if kw in lines[i]:
                    return i
        return len(lines) -1 # Default to before the last line (e.g. tight_layout)

    def _reorder_plot_settings(self, code: str) -> str:
        """Ensure scale settings appear before relevant data or limit settings if added during mutation."""
        lines = code.split('\n')
        # This is a simplified reordering. A full AST parser would be more robust.
        # For now, we assume mutation doesn't drastically misplace scale settings.
        # A simple check: if xscale/yscale is present, move it before set_xlim/ylim/plot if it's after.
        # This part can be complex to do robustly without proper parsing.
        # We will rely on the mutation logic to insert them at reasonable places.
        return "\n".join(lines)


    def _get_current_scales(self, lines: List[str]) -> Tuple[str, str]:
        x_scale = 'linear'
        y_scale = 'linear'
        for line in lines:
            if "ax.set_xscale('log')" in line:
                x_scale = 'log'
            elif "ax.set_yscale('log')" in line:
                y_scale = 'log'
        return x_scale, y_scale

    def _modify_plot_data(self, plot_line: str, all_lines: List[str]) -> str:
        x_scale, y_scale = self._get_current_scales(all_lines)
        x_min_gen, x_max_gen = (0.1, 10) if x_scale == 'log' else (1, 16)
        y_min_gen, y_max_gen = (0.1, 100) if y_scale == 'log' else (0, 20)

        num_points = random.randint(5, 15)
        if x_scale == 'log':
            x_points = sorted([round(random.uniform(x_min_gen, x_max_gen), 2) for _ in range(num_points)])
        else:
            x_points = sorted([round(random.uniform(x_min_gen, x_max_gen), 2) for _ in range(num_points)])

        if y_scale == 'log':
            y_points = [round(random.uniform(y_min_gen, y_max_gen), 2) for _ in range(len(x_points))]
        else:
            y_points = [round(random.uniform(y_min_gen, y_max_gen), 2) for _ in range(len(x_points))]

        # Extract style parameters from original line
        # (This part is kept similar to original for brevity, but could be more robust)
        color, marker, linestyle = self._extract_style_from_line(plot_line)

        return f"ax.plot({x_points}, {y_points}, color='{color}', marker='{marker}', linestyle='{linestyle}', markersize=6, linewidth=2)"

    def _extract_style_from_line(self, plot_line: str) -> Tuple[str, str, str]:
        color = random.choice(self.color_options)
        marker = random.choice(self.marker_options)
        linestyle = random.choice(self.line_styles)
        if 'color=' in plot_line:
            color_start = plot_line.find("color='") + 7
            color_end = plot_line.find("'", color_start)
            color = plot_line[color_start:color_end]
        if 'marker=' in plot_line:
            marker_start = plot_line.find("marker='") + 8
            marker_end = plot_line.find("'", marker_start)
            val = plot_line[marker_start:marker_end]
            marker = val if val != 'None' else None
        if 'linestyle=' in plot_line:
            style_start = plot_line.find("linestyle='") + 11
            style_end = plot_line.find("'", style_start)
            val = plot_line[style_start:style_end]
            linestyle = val if val != 'None' else None
        if marker is None and linestyle is None: linestyle = '-' # Ensure drawable
        return color, marker, linestyle


    def _change_plot_style(self, plot_line: str) -> str:
        try:
            start_bracket = plot_line.find('[')
            if start_bracket == -1: return self._generate_single_curve([]) # Pass empty list for scales

            bracket_count = 0; first_end = -1
            for i, char in enumerate(plot_line[start_bracket:]):
                if char == '[': bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0: first_end = start_bracket + i; break
            if first_end == -1: return self._generate_single_curve([])

            second_start = plot_line.find('[', first_end)
            if second_start == -1: return self._generate_single_curve([])

            bracket_count = 0; second_end = -1
            for i, char in enumerate(plot_line[second_start:]):
                if char == '[': bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0: second_end = second_start + i; break
            if second_end == -1: return self._generate_single_curve([])

            x_data = plot_line[start_bracket:first_end+1]
            y_data = plot_line[second_start:second_end+1]

            color = random.choice(self.color_options)
            marker = random.choice(self.marker_options)
            linestyle = random.choice(self.line_styles)
            if marker is None and linestyle is None: linestyle = '-'

            parts = [f"ax.plot({x_data}, {y_data}"]
            parts.append(f"color='{color}'")
            if marker: parts.append(f"marker='{marker}'")
            if linestyle: parts.append(f"linestyle='{linestyle}'")
            parts.append("markersize=6")
            parts.append("linewidth=2)")
            return ", ".join(parts)

        except Exception:
            return self._generate_single_curve([])

    def _generate_single_curve(self, all_lines: List[str]) -> str:
        x_scale, y_scale = self._get_current_scales(all_lines)
        x_min_gen, x_max_gen = (0.1, 10) if x_scale == 'log' else (1, 16)
        y_min_gen, y_max_gen = (0.1, 100) if y_scale == 'log' else (0, 20)
        num_points = random.randint(5, 15)

        if x_scale == 'log':
            x_points = sorted([round(random.uniform(x_min_gen, x_max_gen), 2) for _ in range(num_points)])
        else:
            x_points = sorted([round(random.uniform(x_min_gen, x_max_gen), 2) for _ in range(num_points)])

        if y_scale == 'log':
            y_points = [round(random.uniform(y_min_gen, y_max_gen), 2) for _ in range(len(x_points))]
        else:
            y_points = [round(random.uniform(y_min_gen, y_max_gen), 2) for _ in range(len(x_points))]

        color = random.choice(self.color_options)
        marker = random.choice(self.marker_options)
        linestyle = random.choice(self.line_styles)
        if marker is None and linestyle is None: linestyle = '-'

        parts = [f"ax.plot({x_points}, {y_points}"]
        parts.append(f"color='{color}'")
        if marker: parts.append(f"marker='{marker}'")
        if linestyle: parts.append(f"linestyle='{linestyle}'")
        parts.append("markersize=6")
        parts.append("linewidth=2)")
        return ", ".join(parts)


class GraphSimilarityEvaluator:
    def __init__(self, target_image_path: str):
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found: {target_image_path}")
        if not target_image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("Target image must be a PNG or JPG file")

        self.target_image_raw = cv2.imread(target_image_path) # Load in color for more features
        if self.target_image_raw is None:
            raise ValueError(f"Could not load image from: {target_image_path}")
        self.target_image_gray = cv2.cvtColor(self.target_image_raw, cv2.COLOR_BGR2GRAY)

        # Pre-calculate features for the target image
        self.target_features = self.extract_features(self.target_image_gray, self.target_image_raw)
        print(f"Loaded target image: {target_image_path} (shape: {self.target_image_gray.shape})")
        print(f"Target features: {self.target_features}")

    def execute_code_and_capture(self, code: str) -> Tuple[np.ndarray, np.ndarray]:
        """Execute matplotlib code and capture the resulting image (gray and color)"""
        try:
            plt.close('all')
            exec_globals = {'plt': plt, 'np': np, 'math': math} # Add math for safety
            exec(code, exec_globals)

            temp_file = 'temp_plot.png'
            plt.savefig(temp_file, dpi=100, bbox_inches='tight')
            plt.close('all')

            if os.path.exists(temp_file):
                img_color = cv2.imread(temp_file)
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                try: os.remove(temp_file)
                except: pass
                return img_gray, img_color
            else:
                return None, None
        except Exception as e:
            # print(f"Error executing code: {e}\nCode:\n{code[:500]}...") # Verbose
            plt.close('all')
            return None, None

    def calculate_similarity(self, generated_gray_image: np.ndarray, generated_color_image: np.ndarray) -> float:
        if generated_gray_image is None or self.target_image_gray is None:
            return 0.0

        try:
            h, w = self.target_image_gray.shape
            generated_resized_gray = cv2.resize(generated_gray_image, (w, h))
            ssim_score = ssim(self.target_image_gray, generated_resized_gray, data_range=generated_resized_gray.max() - generated_resized_gray.min()) # Added data_range

            # Feature-based similarity (rudimentary example)
            generated_features = self.extract_features(generated_gray_image, generated_color_image)
            feature_sim = 0.0
            common_keys = set(self.target_features.keys()).intersection(set(generated_features.keys()))

            if common_keys:
                # Normalized difference for numeric features
                numeric_diff_sum = 0
                count = 0
                for key in ['edge_density', 'num_contours', 'avg_saturation', 'avg_hue', 'num_dominant_colors']:
                    if key in common_keys and isinstance(self.target_features[key], (int, float)):
                        val_target = self.target_features[key]
                        val_gen = generated_features[key]
                        denominator = max(abs(val_target), abs(val_gen), 1e-6) # Avoid division by zero
                        numeric_diff_sum += 1 - (abs(val_target - val_gen) / denominator)
                        count +=1
                feature_sim = (numeric_diff_sum / count) if count > 0 else 0.0
                feature_sim = max(0, min(1, feature_sim)) # clamp to [0,1]

            # Combine SSIM and feature similarity
            # Adjust weights as needed. E.g., 0.7 for SSIM, 0.3 for features
            final_similarity = 0.7 * ssim_score + 0.3 * feature_sim
            return max(0.0, final_similarity)

        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def extract_features(self, image_gray: np.ndarray, image_color: np.ndarray = None) -> Dict:
        if image_gray is None:
            return {}
        features = {}
        try:
            # 1. Edge Density
            edges = cv2.Canny(image_gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size if edges.size > 0 else 0

            # 2. Number of Contours (approximates number of distinct shapes/curves)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features['num_contours'] = len(contours)

            # 3. Brightness
            features['brightness_mean'] = np.mean(image_gray)
            features['brightness_std'] = np.std(image_gray)

            if image_color is not None:
                # 4. Color features (from HSV)
                hsv_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
                features['avg_hue'] = np.mean(hsv_image[:, :, 0])
                features['avg_saturation'] = np.mean(hsv_image[:, :, 1])
                features['avg_value'] = np.mean(hsv_image[:, :, 2]) # Same as brightness_mean for gray

                # 5. Number of dominant colors (simplified)
                # Reshape the image to be a list of pixels
                pixels = image_color.reshape((-1, 3))
                pixels = np.float32(pixels)
                # Define criteria, number of clusters(K) and apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 5 # Look for K dominant colors
                compactness, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                # centers are the K dominant colors in BGR
                # Count how many are significantly present (e.g. >5% of pixels)
                label_counts = np.bincount(labels.flatten())
                significant_colors = sum(1 for count in label_counts if count / len(pixels) > 0.05)
                features['num_dominant_colors'] = significant_colors
            else: # Defaults if no color image
                features['avg_hue'] = 0
                features['avg_saturation'] = 0
                features['avg_value'] = features.get('brightness_mean',0)
                features['num_dominant_colors'] = 0


            # Placeholder for log scale detection (very hard without OCR, often a property of the code)
            # features['is_x_log_visual'] = False # Requires sophisticated analysis
            # features['is_y_log_visual'] = False

        except Exception as e:
            print(f"Error extracting features: {e}")
        return features


class SelfPlayTrainer:
    def __init__(self, target_image_path: str, population_size=30, generations=50): # Reduced defaults for faster testing
        if not target_image_path:
            raise ValueError("Target image path is required")

        self.generator = GraphCodeGenerator()
        self.evaluator = GraphSimilarityEvaluator(target_image_path)
        self.population = []
        self.population_size = population_size
        self.generations_to_run = generations # Store the target number of generations
        self.generation = 0
        self.best_overall_similarity = -1.0
        self.best_overall_code = ""


    def initialize_population(self):
        self.population = []
        successful_individuals = 0
        attempts = 0
        max_attempts = self.population_size * 5 # Increased attempts

        print("Initializing population...")
        while successful_individuals < self.population_size and attempts < max_attempts:
            attempts += 1
            code = self.generator.generate_random_code()
            img_gray, img_color = self.evaluator.execute_code_and_capture(code)

            if img_gray is not None:
                similarity = self.evaluator.calculate_similarity(img_gray, img_color)
                self.population.append({
                    'code': code,
                    'similarity': similarity,
                    'generation': self.generation
                })
                successful_individuals += 1
                if successful_individuals % (self.population_size // 10 + 1) == 0:
                    print(f"  Initialized {successful_individuals}/{self.population_size} individuals...")

        if not self.population:
            raise Exception("Failed to generate any valid individuals during initialization.")

        self.population.sort(key=lambda x: x['similarity'], reverse=True)
        if self.population:
            current_best_sim = self.population[0]['similarity']
            if current_best_sim > self.best_overall_similarity:
                self.best_overall_similarity = current_best_sim
                self.best_overall_code = self.population[0]['code']
        print(f"Initialization complete. Population size: {len(self.population)}. Best initial similarity: {self.best_overall_similarity:.4f}")


    def evolve_generation(self):
        self.generation += 1
        if not self.population:
            print("Warning: Population is empty. Re-initializing.")
            self.initialize_population()
            if not self.population: # Still empty
                print("Critical: Failed to re-initialize population. Stopping.")
                return False # Indicate failure to evolve


        elite_size = max(1, self.population_size // 5) # Slightly larger elite
        new_population = [ind.copy() for ind in self.population[:elite_size]]

        # Offspring generation
        num_offspring_needed = self.population_size - elite_size
        offspring_generated = 0
        attempts = 0
        max_mutation_attempts = num_offspring_needed * 5 # Try harder to generate offspring

        while offspring_generated < num_offspring_needed and attempts < max_mutation_attempts:
            attempts += 1
            # Tournament selection for parent
            parent1 = random.choice(self.population[:max(1, self.population_size // 2)])
            # Parent2 = random.choice(self.population[:max(1, self.population_size // 2)]) # For crossover later

            # Primarily mutation, occasionally new random individual
            if random.random() < 0.9 and parent1: # 90% mutation
                child_code = self.generator.mutate_code(parent1['code'])
            else: # 10% new random
                child_code = self.generator.generate_random_code()

            img_gray, img_color = self.evaluator.execute_code_and_capture(child_code)
            if img_gray is not None:
                similarity = self.evaluator.calculate_similarity(img_gray, img_color)
                new_population.append({
                    'code': child_code,
                    'similarity': similarity,
                    'generation': self.generation
                })
                offspring_generated += 1

        if not new_population: # If elite somehow failed or no offspring
             print(f"Generation {self.generation}: Failed to generate any valid individuals for new population. Retrying with random.")
             self.initialize_population() # Fallback to re-initializing
             return True if self.population else False


        self.population = sorted(new_population, key=lambda x: x['similarity'], reverse=True)[:self.population_size] # Ensure pop size

        if self.population:
            best_current_gen_similarity = self.population[0]['similarity']
            if best_current_gen_similarity > self.best_overall_similarity:
                self.best_overall_similarity = best_current_gen_similarity
                self.best_overall_code = self.population[0]['code']
                print(f"Generation {self.generation}: New best overall similarity = {self.best_overall_similarity:.4f} (Pop: {len(self.population)}) *")
            else:
                print(f"Generation {self.generation}: Best this gen = {best_current_gen_similarity:.4f}, Overall best = {self.best_overall_similarity:.4f} (Pop: {len(self.population)})")
        else:
            print(f"Generation {self.generation}: Population became empty. This should not happen.")
            return False
        return True


    def train(self, generations: int = None): # Use stored generations if None
        if generations is None:
            generations = self.generations_to_run

        print(f"Starting training for {generations} generations...")
        self.initialize_population()

        if not self.population:
            print("Training stopped: Initial population is empty.")
            return

        print(f"Initial best overall similarity: {self.best_overall_similarity:.4f}")

        for gen_num in range(1, generations + 1):
            if not self.evolve_generation():
                print(f"Stopping training at generation {self.generation} due to critical error.")
                break
            if gen_num % 10 == 0 or gen_num == generations:
                self.save_best_result(f"best_gen_{self.generation}_sim_{self.best_overall_similarity:.4f}.py")
                # Also save a preview of the best image
                if self.best_overall_code:
                    img_gray, _ = self.evaluator.execute_code_and_capture(self.best_overall_code)
                    if img_gray is not None:
                        cv2.imwrite(f"best_gen_{self.generation}_sim_{self.best_overall_similarity:.4f}.png", img_gray)


        print(f"Training finished after {self.generation} generations.")
        print(f"Best overall similarity achieved: {self.best_overall_similarity:.4f}")


    def save_best_result(self, filename: str):
        if self.best_overall_code:
            try:
                # Sanitize filename slightly for similarity scores
                filename = filename.replace("..", ".").replace(":", "-")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# Best Similarity: {self.best_overall_similarity:.4f}\n")
                    f.write(f"# Generation: {self.generation}\n")
                    f.write(self.best_overall_code)
                print(f"Saved best result to {filename}")
            except Exception as e:
                print(f"Error saving file: {e}")
        else:
            print("No best code available to save.")

    def get_best_code(self) -> str:
        return self.best_overall_code


def demonstrate_system(target_image_file: str, generations=20, population_size=20): # Smaller defaults for demo
    """Demonstrate the self-play system"""
    if not os.path.exists(target_image_file):
        print(f"ERROR: Target image '{target_image_file}' not found. Please provide a valid path.")
        # Create a dummy PNG if it doesn't exist for demonstration purposes
        print("Creating a dummy black PNG as a fallback target for demonstration.")
        dummy_img = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite("dummy_target.png", dummy_img)
        target_image_file = "dummy_target.png"


    print(f"Running self-play training with target image: {target_image_file}")
    trainer = SelfPlayTrainer(target_image_path=target_image_file, population_size=population_size, generations=generations)

    try:
        trainer.train() # Uses generations passed to constructor
        best_code = trainer.get_best_code()
        print("\n" + "=" * 50)
        print("Best generated code overall:")
        print("=" * 50)
        print(best_code)
        print(f"Achieved similarity: {trainer.best_overall_similarity:.4f}")

        if best_code:
            print("\n" + "=" * 50)
            print("Testing (displaying) best generated code...")
            try:
                plt.close('all')
                exec_globals = {'plt': plt, 'np': np, 'math': math}
                exec(best_code, exec_globals)
                plt.suptitle(f"Best Generated Plot (Sim: {trainer.best_overall_similarity:.4f})", fontsize=10)
                plt.show() # Display the plot
                print("Code executed and displayed successfully!")
            except Exception as e:
                print(f"Error displaying best code: {e}")
        else:
            print("No best code was generated.")

    except Exception as e:
        print(f"Training or demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # IMPORTANT: Replace with the actual paths to your images
    # You need to have 'image_b2f98e.png' and 'image_b2f972.png' in the same directory
    # or provide full paths.

    target_image_2 = 'Output_2008\extracted_graphs\page_002_graph_2.png' # First image you provided
    target_image_1 = 'Output_2008\extracted_graphs\page_001_graph_1.png' # Second image you provided

    # Create dummy images if they don't exist, so the script can run.
    # You should replace these with your actual image files.
    for img_path in [target_image_1, target_image_2]:
        if not os.path.exists(img_path):
            print(f"Warning: Target image '{img_path}' not found. Creating a dummy black PNG.")
            print("Please replace with your actual image file for meaningful results.")
            dummy_array = np.zeros((480, 640), dtype=np.uint8) # Typical plot size
            cv2.imwrite(img_path, dummy_array)


    print("\n--- Demonstrating with First Target Image ---")
    demonstrate_system(target_image_file=target_image_1, generations=30, population_size=20) # Reduced for quicker demo

    # print("\n--- Demonstrating with Second Target Image ---")
    # demonstrate_system(target_image_file=target_image_2, generations=30, population_size=20)