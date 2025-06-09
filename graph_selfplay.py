import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Tuple, Dict
import json
import os

class GraphCodeGenerator:
    """Generates Python code for matplotlib plots"""
    
    def __init__(self):
        # Use single-letter color codes for matplotlib compatibility
        self.color_options = ['r', 'b', 'g', 'k', 'c', 'm', 'y']
        self.marker_options = ['o', 's', '^', 'v', '<', '>', 'D', 'x', '+']
        self.line_styles = ['-', '--', '-.', ':']
    
    def generate_random_code(self) -> str:
        """Generate random matplotlib code"""
        num_curves = random.randint(1, 5)
        
        code_parts = [
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "fig, ax = plt.subplots(figsize=(8, 6))"
        ]
        
        for i in range(num_curves):
            # Generate random data points
            x_points = sorted([round(random.uniform(1, 16), 2) for _ in range(random.randint(5, 12))])
            y_points = [round(random.uniform(0, 20), 2) for _ in range(len(x_points))]
            
            color = random.choice(self.color_options)
            marker = random.choice(self.marker_options)
            linestyle = random.choice(self.line_styles)
            
            # Format as proper Python lists
            x_str = str(x_points)
            y_str = str(y_points)
            
            # Create separate style parameters to avoid format string issues
            plot_line = f"ax.plot({x_str}, {y_str}, color='{color}', marker='{marker}', linestyle='{linestyle}', markersize=6, linewidth=2)"
            code_parts.append(plot_line)
        
        # Add formatting
        code_parts.extend([
            "ax.set_xlabel('Density (g cm^-3)', fontsize=12)",
            "ax.set_ylabel('Electronic energy gap (eV)', fontsize=12)",
            "ax.set_xlim(1, 16)",
            "ax.set_ylim(0, 20)",
            "ax.grid(True, alpha=0.3)",
            "plt.tight_layout()"
        ])
        
        return "\n".join(code_parts)
    
    def mutate_code(self, code: str) -> str:
        """Mutate existing code to create variations"""
        lines = code.split('\n')
        plot_lines = [i for i, line in enumerate(lines) if 'ax.plot(' in line]
        
        if not plot_lines:
            return self.generate_random_code()
        
        # Choose mutation type
        mutation_type = random.choice(['modify_data', 'change_style', 'add_curve', 'remove_curve'])
        
        if mutation_type == 'modify_data' and plot_lines:
            # Modify data points in existing curve
            line_idx = random.choice(plot_lines)
            lines[line_idx] = self._modify_plot_data(lines[line_idx])
            
        elif mutation_type == 'change_style' and plot_lines:
            # Change color/marker/style
            line_idx = random.choice(plot_lines)
            lines[line_idx] = self._change_plot_style(lines[line_idx])
            
        elif mutation_type == 'add_curve':
            # Add new curve
            new_curve = self._generate_single_curve()
            lines.insert(-4, new_curve)  # Insert before formatting lines
            
        elif mutation_type == 'remove_curve' and len(plot_lines) > 1:
            # Remove a curve
            line_idx = random.choice(plot_lines)
            lines.pop(line_idx)
        
        return "\n".join(lines)
    
    def _modify_plot_data(self, plot_line: str) -> str:
        """Modify data points in a plot line"""
        # Generate new data
        x_points = sorted([round(random.uniform(1, 16), 2) for _ in range(random.randint(5, 12))])
        y_points = [round(random.uniform(0, 20), 2) for _ in range(len(x_points))]
        
        # Extract style parameters from original line
        if 'color=' in plot_line:
            color_start = plot_line.find("color='") + 7
            color_end = plot_line.find("'", color_start)
            color = plot_line[color_start:color_end]
        else:
            color = random.choice(self.color_options)
            
        if 'marker=' in plot_line:
            marker_start = plot_line.find("marker='") + 8
            marker_end = plot_line.find("'", marker_start)
            marker = plot_line[marker_start:marker_end]
        else:
            marker = random.choice(self.marker_options)
            
        if 'linestyle=' in plot_line:
            style_start = plot_line.find("linestyle='") + 11
            style_end = plot_line.find("'", style_start)
            linestyle = plot_line[style_start:style_end]
        else:
            linestyle = random.choice(self.line_styles)
        
        return f"ax.plot({x_points}, {y_points}, color='{color}', marker='{marker}', linestyle='{linestyle}', markersize=6, linewidth=2)"
    
    def _change_plot_style(self, plot_line: str) -> str:
        """Change the style of a plot line"""
        # Extract data points from original line
        try:
            # Find the data arrays in the line
            start_bracket = plot_line.find('[')
            if start_bracket == -1:
                return self._generate_single_curve()
                
            # Find first closing bracket
            bracket_count = 0
            first_end = -1
            for i, char in enumerate(plot_line[start_bracket:]):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        first_end = start_bracket + i
                        break
            
            if first_end == -1:
                return self._generate_single_curve()
                
            # Find second array
            second_start = plot_line.find('[', first_end)
            if second_start == -1:
                return self._generate_single_curve()
                
            bracket_count = 0
            second_end = -1
            for i, char in enumerate(plot_line[second_start:]):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        second_end = second_start + i
                        break
            
            if second_end == -1:
                return self._generate_single_curve()
                
            x_data = plot_line[start_bracket:first_end+1]
            y_data = plot_line[second_start:second_end+1]
            
            # Generate new style
            color = random.choice(self.color_options)
            marker = random.choice(self.marker_options)
            linestyle = random.choice(self.line_styles)
            
            return f"ax.plot({x_data}, {y_data}, color='{color}', marker='{marker}', linestyle='{linestyle}', markersize=6, linewidth=2)"
            
        except Exception:
            return self._generate_single_curve()
    
    def _generate_single_curve(self) -> str:
        """Generate a single curve line"""
        x_points = sorted([round(random.uniform(1, 16), 2) for _ in range(random.randint(5, 12))])
        y_points = [round(random.uniform(0, 20), 2) for _ in range(len(x_points))]
        
        color = random.choice(self.color_options)
        marker = random.choice(self.marker_options)
        linestyle = random.choice(self.line_styles)
        
        return f"ax.plot({x_points}, {y_points}, color='{color}', marker='{marker}', linestyle='{linestyle}', markersize=6, linewidth=2)"

class GraphSimilarityEvaluator:
    """Evaluate similarity between generated graph and target"""
    
    def __init__(self, target_image_path: str):
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found: {target_image_path}")
        
        if not target_image_path.lower().endswith('.png'):
            raise ValueError("Target image must be a PNG file")
            
        self.target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
        if self.target_image is None:
            raise ValueError(f"Could not load image from: {target_image_path}")
        
        print(f"Loaded target image: {target_image_path} (shape: {self.target_image.shape})")
    
    def execute_code_and_capture(self, code: str) -> np.ndarray:
        """Execute matplotlib code and capture the resulting image"""
        try:
            # Close any existing figures to prevent memory issues
            plt.close('all')
            
            # Execute the code safely
            exec_globals = {'plt': plt, 'np': np}
            exec(code, exec_globals)
            
            # Save to buffer and convert to image
            temp_file = 'temp_plot.png'
            plt.savefig(temp_file, dpi=100, bbox_inches='tight')
            plt.close('all')
            
            # Load and return image
            if os.path.exists(temp_file):
                img = cv2.imread(temp_file, cv2.IMREAD_GRAYSCALE)
                # Clean up temp file
                try:
                    os.remove(temp_file)
                except:
                    pass
                return img
            else:
                return None
                
        except Exception as e:
            print(f"Error executing code: {e}")
            plt.close('all')
            return None
    
    def calculate_similarity(self, generated_image: np.ndarray, target_image: np.ndarray = None) -> float:
        """Calculate similarity between generated and target images"""
        if target_image is None:
            target_image = self.target_image
        
        if generated_image is None or target_image is None:
            return 0.0
        
        try:
            # Resize images to same size
            h, w = target_image.shape
            generated_resized = cv2.resize(generated_image, (w, h))
            
            # Calculate SSIM (Structural Similarity Index)
            similarity = ssim(target_image, generated_resized)
            
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """Extract features from graph image for comparison"""
        if image is None:
            return {}
        
        features = {}
        
        try:
            # Edge detection to find plot lines
            edges = cv2.Canny(image, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # Detect potential data points (circles, markers)
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=1, maxRadius=10)
            features['num_markers'] = len(circles[0]) if circles is not None else 0
            
            # Color analysis
            features['brightness_mean'] = np.mean(image)
            features['brightness_std'] = np.std(image)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
        
        return features

class SelfPlayTrainer:
    """Self-play trainer for graph code generation"""
    
    def __init__(self, target_image_path: str):
        if not target_image_path:
            raise ValueError("Target image path is required")
        
        self.generator = GraphCodeGenerator()
        self.evaluator = GraphSimilarityEvaluator(target_image_path)
        self.population = []
        self.population_size = 50
        self.generation = 0
        
    def initialize_population(self):
        """Initialize random population"""
        self.population = []
        successful_individuals = 0
        attempts = 0
        max_attempts = self.population_size * 3
        
        while successful_individuals < self.population_size and attempts < max_attempts:
            attempts += 1
            code = self.generator.generate_random_code()
            img = self.evaluator.execute_code_and_capture(code)
            
            if img is not None:
                similarity = self.evaluator.calculate_similarity(img)
                
                self.population.append({
                    'code': code,
                    'similarity': similarity,
                    'generation': self.generation
                })
                successful_individuals += 1
        
        if len(self.population) == 0:
            raise Exception("Failed to generate any valid individuals")
        
        # Sort by similarity
        self.population.sort(key=lambda x: x['similarity'], reverse=True)
    
    def evolve_generation(self):
        """Evolve to next generation using self-play"""
        self.generation += 1
        
        # Keep top performers
        elite_size = max(1, self.population_size // 4)
        new_population = [ind.copy() for ind in self.population[:elite_size]]
        
        # Generate offspring through mutation and crossover
        attempts = 0
        max_attempts = self.population_size * 3
        
        while len(new_population) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            if random.random() < 0.8 and len(self.population) > 0:  # Mutation
                parent = random.choice(self.population[:max(1, self.population_size//2)])
                child_code = self.generator.mutate_code(parent['code'])
            else:  # Generate new random
                child_code = self.generator.generate_random_code()
            
            # Evaluate child
            img = self.evaluator.execute_code_and_capture(child_code)
            if img is not None:
                similarity = self.evaluator.calculate_similarity(img)
                
                new_population.append({
                    'code': child_code,
                    'similarity': similarity,
                    'generation': self.generation
                })
        
        if len(new_population) > 0:
            # Update population
            self.population = sorted(new_population, key=lambda x: x['similarity'], reverse=True)
            
            # Print progress
            best_similarity = self.population[0]['similarity']
            print(f"Generation {self.generation}: Best similarity = {best_similarity:.4f} (Population: {len(self.population)})")
        else:
            print(f"Generation {self.generation}: Failed to generate valid offspring")
    
    def train(self, generations: int = 50):
        """Train the system for specified generations"""
        print("Initializing population...")
        self.initialize_population()
        
        print(f"Initial best similarity: {self.population[0]['similarity']:.4f}")
        
        for gen in range(generations):
            self.evolve_generation()
            
            # Save best result periodically
            if gen % 10 == 0:
                self.save_best_result(f"best_gen_{gen}.py")
    
    def save_best_result(self, filename: str):
        """Save the best code result"""
        if len(self.population) > 0:
            best_code = self.population[0]['code']
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(best_code)
                print(f"Saved best result to {filename}")
            except Exception as e:
                print(f"Error saving file: {e}")
    
    def get_best_code(self) -> str:
        """Get the best generated code"""
        if len(self.population) > 0:
            return self.population[0]['code']
        return ""

# Example usage and demonstration
def demonstrate_system():
    """Demonstrate the self-play system"""
    
    # Create sample target code that matches your graph
    target_code = '''
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))

# Black curve (highest energy)
x1 = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
y1 = [18.5, 16.2, 15.8, 12.1, 10.8, 9.2, 7.8, 6.1, 4.2, 2.8, 0.5, 0.2, 0.1, 0.05]

# Blue curve
x2 = [1, 2, 3, 4, 5, 6, 7, 8]
y2 = [9.8, 7.2, 4.1, 3.2, 2.1, 1.2, 0.5, 0.1]

# Green curve
x3 = [1, 2, 3, 4, 5, 6]
y3 = [7.2, 4.8, 2.1, 1.8, 0.8, 0.2]

# Red curve (lowest)
x4 = [1, 2, 3, 4]
y4 = [3.8, 1.2, 0.5, 0.1]

ax.plot(x1, y1, color='k', marker='o', linestyle='-', markersize=6, linewidth=2, label='Series 1')
ax.plot(x2, y2, color='b', marker='o', linestyle='-', markersize=6, linewidth=2, label='Series 2')
ax.plot(x3, y3, color='g', marker='o', linestyle='-', markersize=6, linewidth=2, label='Series 3')
ax.plot(x4, y4, color='r', marker='o', linestyle='-', markersize=6, linewidth=2, label='Series 4')

ax.set_xlabel('Density (g cm^-3)', fontsize=12)
ax.set_ylabel('Electronic energy gap (eV)', fontsize=12)
ax.set_xlim(1, 16)
ax.set_ylim(0, 20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''
    
    print("Target code example:")
    print("=" * 50)
    
    # Test without target image first
    #print("Running self-play training without target image...")
    trainer = SelfPlayTrainer('Output_2008\extracted_graphs\page_001_graph_1.png')
    
    try:
        trainer.train(generations=50)
        best_code = trainer.get_best_code()
        print("\n" + "=" * 50)
        print("Best generated code:")
        print("=" * 50)
        print(best_code)
        
        # Test the best code
        print("\n" + "=" * 50)
        print("Testing best generated code...")
        img = trainer.evaluator.execute_code_and_capture(best_code)
        if img is not None:
            print("Code executed successfully!")
        else:
            print("Code execution failed.")
            
    except Exception as e:
        print(f"Training failed: {e}")
        print("\nTrying with sample target code instead...")
        
        # Show the target code as an example
        exec_globals = {'plt': plt, 'np': np}
        try:
            exec(target_code, exec_globals)
        except Exception as e:
            print(f"Error executing target code: {e}")

if __name__ == "__main__":
    demonstrate_system()