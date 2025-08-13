"""
Diagram-as-Code Absolute Zero Reasoner (DAC-AZR)
A system for generating synthetic datasets of diagrams and Python programs using self-play.

This implementation follows the Absolute Zero paradigm:
1. PROPOSE: Generate diagram concepts and Python programs
2. SOLVE: Verify and correct the generated programs
3. Self-play loop for continuous improvement
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import json
import os
import random
import ast
import re
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiagramSpec:
    """Semantic representation of a diagram"""
    diagram_type: str  # 'line_plot', 'bar_chart', 'scatter', 'pie_chart', 'heatmap'
    elements: List[Dict[str, Any]]  # List of plot elements with properties
    properties: Dict[str, Any]  # Global properties like axes, labels, etc.
    complexity_score: float  # Difficulty/complexity rating
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiagramSpec':
        return cls(**data)

@dataclass
class CodeProgram:
    """A Python program that generates a diagram"""
    code: str
    diagram_spec: DiagramSpec
    success_score: float  # How well it matches the target
    complexity_score: float  # Program complexity
    generation_metadata: Dict[str, Any]  # Generation info
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeProgram':
        return cls(**data)

class DiagramVerifier(ABC):
    """Abstract base class for diagram verification"""
    
    @abstractmethod
    def verify_program(self, code: str, target_spec: DiagramSpec) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify if a Python program generates the target diagram
        
        Returns:
            Tuple of (success, score, details)
        """
        pass

class SemanticDiagramVerifier(DiagramVerifier):
    """Verifies diagrams using semantic comparison rather than visual comparison"""
    
    def __init__(self):
        self.supported_types = {
            'line_plot': self._verify_line_plot,
            'bar_chart': self._verify_bar_chart,
            'scatter': self._verify_scatter,
            'pie_chart': self._verify_pie_chart,
            'heatmap': self._verify_heatmap
        }
    
    def verify_program(self, code: str, target_spec: DiagramSpec) -> Tuple[bool, float, Dict[str, Any]]:
        """Verify program using semantic analysis"""
        try:
            # Execute the code safely
            exec_globals = {'plt': plt, 'np': np, 'patches': patches}
            exec(code, exec_globals)
            
            # Get the current figure
            fig = plt.gcf()
            if not fig:
                return False, 0.0, {'error': 'No figure generated'}
            
            # Extract semantic information from the generated plot
            generated_spec = self._extract_plot_semantics(fig)
            
            # Compare with target specification
            success, score, details = self._compare_specs(target_spec, generated_spec)
            
            # Clean up
            plt.close(fig)
            
            return success, score, details
            
        except Exception as e:
            return False, 0.0, {'error': str(e)}
    
    def _extract_plot_semantics(self, fig) -> DiagramSpec:
        """Extract semantic information from a matplotlib figure"""
        ax = fig.axes[0] if fig.axes else None
        if not ax:
            return DiagramSpec('unknown', [], {}, 0.0)
        
        elements = []
        properties = {}
        
        # Extract plot elements
        for line in ax.lines:
            elements.append({
                'type': 'line',
                'x_data': line.get_xdata().tolist(),
                'y_data': line.get_ydata().tolist(),
                'color': line.get_color(),
                'linestyle': line.get_linestyle(),
                'marker': line.get_marker()
            })
        
        # Extract bar elements
        for patch in ax.patches:
            if isinstance(patch, patches.Rectangle):
                elements.append({
                    'type': 'bar',
                    'x': patch.get_x(),
                    'y': patch.get_y(),
                    'width': patch.get_width(),
                    'height': patch.get_height(),
                    'color': patch.get_facecolor()
                })
        
        # Extract properties
        properties = {
            'xlabel': ax.get_xlabel(),
            'ylabel': ax.get_ylabel(),
            'title': ax.get_title(),
            'xlim': ax.get_xlim(),
            'ylim': ax.get_ylim(),
            'grid': ax.get_xgrid()
        }
        
        # Determine diagram type
        diagram_type = self._infer_diagram_type(elements)
        
        # Calculate complexity score
        complexity = self._calculate_complexity(elements, properties)
        
        return DiagramSpec(diagram_type, elements, properties, complexity)
    
    def _infer_diagram_type(self, elements: List[Dict]) -> str:
        """Infer diagram type from elements"""
        if any(e['type'] == 'bar' for e in elements):
            return 'bar_chart'
        elif any(e['type'] == 'line' for e in elements):
            return 'line_plot'
        else:
            return 'scatter'
    
    def _calculate_complexity(self, elements: List[Dict], properties: Dict) -> float:
        """Calculate complexity score based on elements and properties"""
        base_complexity = len(elements) * 0.2
        
        # Add complexity for data points
        for element in elements:
            if 'x_data' in element:
                base_complexity += len(element['x_data']) * 0.01
        
        # Add complexity for properties
        if properties.get('grid'):
            base_complexity += 0.1
        if properties.get('title'):
            base_complexity += 0.1
            
        return min(base_complexity, 1.0)
    
    def _compare_specs(self, target: DiagramSpec, generated: DiagramSpec) -> Tuple[bool, float, Dict[str, Any]]:
        """Compare target and generated specifications"""
        if target.diagram_type != generated.diagram_type:
            return False, 0.0, {'type_mismatch': f'Expected {target.diagram_type}, got {generated.diagram_type}'}
        
        # Compare elements
        element_score = self._compare_elements(target.elements, generated.elements)
        
        # Compare properties
        property_score = self._compare_properties(target.properties, generated.properties)
        
        # Overall score
        overall_score = (element_score + property_score) / 2
        
        success = overall_score > 0.7  # Threshold for success
        
        details = {
            'element_score': element_score,
            'property_score': property_score,
            'overall_score': overall_score
        }
        
        return success, overall_score, details
    
    def _compare_elements(self, target_elements: List[Dict], generated_elements: List[Dict]) -> float:
        """Compare plot elements"""
        if len(target_elements) != len(generated_elements):
            return 0.5  # Partial credit for partial match
        
        total_score = 0.0
        for target_elem, gen_elem in zip(target_elements, generated_elements):
            if target_elem['type'] != gen_elem['type']:
                continue
            
            if target_elem['type'] == 'line':
                # Compare data points (simplified comparison)
                if 'x_data' in target_elem and 'x_data' in gen_elem:
                    x_similarity = self._compare_arrays(target_elem['x_data'], gen_elem['x_data'])
                    y_similarity = self._compare_arrays(target_elem['y_data'], gen_elem['y_data'])
                    total_score += (x_similarity + y_similarity) / 2
                else:
                    total_score += 0.5  # Partial credit for type match
            
            elif target_elem['type'] == 'bar':
                # Compare bar properties
                if abs(target_elem.get('height', 0) - gen_elem.get('height', 0)) < 0.1:
                    total_score += 1.0
                else:
                    total_score += 0.5
        
        return total_score / len(target_elements) if target_elements else 0.0
    
    def _compare_arrays(self, arr1: List, arr2: List) -> float:
        """Compare two arrays for similarity"""
        if len(arr1) != len(arr2):
            return 0.0
        
        # Simple correlation-based similarity
        try:
            correlation = np.corrcoef(arr1, arr2)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _compare_properties(self, target_props: Dict, generated_props: Dict) -> float:
        """Compare plot properties"""
        score = 0.0
        total_props = 0
        
        for key in target_props:
            if key in generated_props:
                if key in ['xlabel', 'ylabel', 'title']:
                    # Text similarity
                    if target_props[key] == generated_props[key]:
                        score += 1.0
                    else:
                        score += 0.5  # Partial credit for having the property
                elif key in ['xlim', 'ylim']:
                    # Range similarity
                    target_range = target_props[key]
                    gen_range = generated_props[key]
                    if len(target_range) == 2 and len(gen_range) == 2:
                        range_similarity = 1.0 - min(1.0, abs(target_range[0] - gen_range[0]) + abs(target_range[1] - gen_range[1]))
                        score += range_similarity
                else:
                    # Boolean or other properties
                    if target_props[key] == generated_props[key]:
                        score += 1.0
                    else:
                        score += 0.5
                total_props += 1
        
        return score / total_props if total_props > 0 else 1.0
    
    # Specific verification methods for different diagram types
    def _verify_line_plot(self, code: str, target_spec: DiagramSpec) -> Tuple[bool, float, Dict[str, Any]]:
        return self.verify_program(code, target_spec)
    
    def _verify_bar_chart(self, code: str, target_spec: DiagramSpec) -> Tuple[bool, float, Dict[str, Any]]:
        return self.verify_program(code, target_spec)
    
    def _verify_scatter(self, code: str, target_spec: DiagramSpec) -> Tuple[bool, float, Dict[str, Any]]:
        return self.verify_program(code, target_spec)
    
    def _verify_pie_chart(self, code: str, target_spec: DiagramSpec) -> Tuple[bool, float, Dict[str, Any]]:
        return self.verify_program(code, target_spec)
    
    def _verify_heatmap(self, code: str, target_spec: DiagramSpec) -> Tuple[bool, float, Dict[str, Any]]:
        return self.verify_program(code, target_spec)

class DiagramProposer:
    """Generates diagram concepts and Python programs (PROPOSE role)"""
    
    def __init__(self, complexity_range: Tuple[float, float] = (0.1, 1.0)):
        self.complexity_range = complexity_range
        self.verifier = SemanticDiagramVerifier()
        self.generated_concepts = set()  # Track diversity
        
    def propose_diagram_concept(self) -> DiagramSpec:
        """Generate a new diagram concept"""
        diagram_type = random.choice(['line_plot', 'bar_chart', 'scatter', 'pie_chart'])
        
        if diagram_type == 'line_plot':
            return self._generate_line_plot_concept()
        elif diagram_type == 'bar_chart':
            return self._generate_bar_chart_concept()
        elif diagram_type == 'scatter':
            return self._generate_scatter_concept()
        elif diagram_type == 'pie_chart':
            return self._generate_pie_chart_concept()
        else:
            return self._generate_line_plot_concept()
    
    def _generate_line_plot_concept(self) -> DiagramSpec:
        """Generate a line plot concept"""
        num_lines = random.randint(1, 4)
        elements = []
        
        for i in range(num_lines):
            num_points = random.randint(5, 15)
            x_data = sorted([round(random.uniform(0, 10), 2) for _ in range(num_points)])
            y_data = [round(random.uniform(0, 20), 2) for _ in range(num_points)]
            
            elements.append({
                'type': 'line',
                'x_data': x_data,
                'y_data': y_data,
                'color': random.choice(['r', 'b', 'g', 'k', 'c', 'm']),
                'linestyle': random.choice(['-', '--', '-.', ':']),
                'marker': random.choice(['o', 's', '^', 'v', 'D', 'x'])
            })
        
        properties = {
            'xlabel': random.choice(['Time (s)', 'Distance (m)', 'Temperature (K)', 'Pressure (Pa)']),
            'ylabel': random.choice(['Amplitude', 'Energy (J)', 'Velocity (m/s)', 'Force (N)']),
            'title': random.choice(['Experimental Data', 'Simulation Results', 'Trend Analysis', '']),
            'xlim': (0, 10),
            'ylim': (0, 20),
            'grid': random.choice([True, False])
        }
        
        complexity = self._calculate_concept_complexity(elements, properties)
        
        return DiagramSpec('line_plot', elements, properties, complexity)
    
    def _generate_bar_chart_concept(self) -> DiagramSpec:
        """Generate a bar chart concept"""
        num_bars = random.randint(3, 8)
        elements = []
        
        for i in range(num_bars):
            elements.append({
                'type': 'bar',
                'x': i,
                'y': 0,
                'width': 0.8,
                'height': round(random.uniform(1, 20), 2),
                'color': random.choice(['r', 'b', 'g', 'k', 'c', 'm'])
            })
        
        properties = {
            'xlabel': 'Categories',
            'ylabel': 'Values',
            'title': random.choice(['Data Comparison', 'Survey Results', 'Performance Metrics', '']),
            'xlim': (-0.5, num_bars - 0.5),
            'ylim': (0, 25),
            'grid': random.choice([True, False])
        }
        
        complexity = self._calculate_concept_complexity(elements, properties)
        
        return DiagramSpec('bar_chart', elements, properties, complexity)
    
    def _generate_scatter_concept(self) -> DiagramSpec:
        """Generate a scatter plot concept"""
        num_points = random.randint(10, 50)
        x_data = [round(random.uniform(0, 10), 2) for _ in range(num_points)]
        y_data = [round(random.uniform(0, 20), 2) for _ in range(num_points)]
        
        elements = [{
            'type': 'scatter',
            'x_data': x_data,
            'y_data': y_data,
            'color': random.choice(['r', 'b', 'g', 'k', 'c', 'm']),
            'marker': random.choice(['o', 's', '^', 'v', 'D', 'x'])
        }]
        
        properties = {
            'xlabel': random.choice(['X Variable', 'Input', 'Feature 1', 'Time']),
            'ylabel': random.choice(['Y Variable', 'Output', 'Feature 2', 'Response']),
            'title': random.choice(['Data Distribution', 'Correlation Analysis', 'Scatter Plot', '']),
            'xlim': (0, 10),
            'ylim': (0, 20),
            'grid': random.choice([True, False])
        }
        
        complexity = self._calculate_concept_complexity(elements, properties)
        
        return DiagramSpec('scatter', elements, properties, complexity)
    
    def _generate_pie_chart_concept(self) -> DiagramSpec:
        """Generate a pie chart concept"""
        num_slices = random.randint(3, 6)
        elements = []
        
        total = 100
        remaining = total
        for i in range(num_slices - 1):
            slice_value = round(random.uniform(5, remaining - (num_slices - i - 1) * 5), 1)
            elements.append({
                'type': 'pie_slice',
                'value': slice_value,
                'label': f'Category {i+1}',
                'color': random.choice(['r', 'b', 'g', 'k', 'c', 'm'])
            })
            remaining -= slice_value
        
        # Last slice gets remaining value
        elements.append({
            'type': 'pie_slice',
            'value': round(remaining, 1),
            'label': f'Category {num_slices}',
            'color': random.choice(['r', 'b', 'g', 'k', 'c', 'm'])
        })
        
        properties = {
            'title': random.choice(['Distribution', 'Composition', 'Breakdown', '']),
            'grid': False
        }
        
        complexity = self._calculate_concept_complexity(elements, properties)
        
        return DiagramSpec('pie_chart', elements, properties, complexity)
    
    def _calculate_concept_complexity(self, elements: List[Dict], properties: Dict) -> float:
        """Calculate complexity of a diagram concept"""
        base_complexity = len(elements) * 0.2
        
        # Add complexity for data points
        for element in elements:
            if 'x_data' in element:
                base_complexity += len(element['x_data']) * 0.01
        
        # Add complexity for properties
        if properties.get('grid'):
            base_complexity += 0.1
        if properties.get('title'):
            base_complexity += 0.1
            
        return min(base_complexity, 1.0)
    
    def generate_python_program(self, diagram_spec: DiagramSpec) -> str:
        """Generate Python code for a diagram specification"""
        if diagram_spec.diagram_type == 'line_plot':
            return self._generate_line_plot_code(diagram_spec)
        elif diagram_spec.diagram_type == 'bar_chart':
            return self._generate_bar_chart_code(diagram_spec)
        elif diagram_spec.diagram_type == 'scatter':
            return self._generate_scatter_code(diagram_spec)
        elif diagram_spec.diagram_type == 'pie_chart':
            return self._generate_pie_chart_code(diagram_spec)
        else:
            return self._generate_line_plot_code(diagram_spec)
    
    def _generate_line_plot_code(self, spec: DiagramSpec) -> str:
        """Generate code for line plot"""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "fig, ax = plt.subplots(figsize=(8, 6))"
        ]
        
        for i, element in enumerate(spec.elements):
            x_data = element['x_data']
            y_data = element['y_data']
            color = element['color']
            linestyle = element['linestyle']
            marker = element['marker']
            
            code_lines.append(f"ax.plot({x_data}, {y_data}, color='{color}', linestyle='{linestyle}', marker='{marker}', markersize=6, linewidth=2)")
        
        # Add properties
        if spec.properties.get('title'):
            code_lines.append(f"ax.set_title('{spec.properties['title']}', fontsize=12)")
        if spec.properties.get('xlabel'):
            code_lines.append(f"ax.set_xlabel('{spec.properties['xlabel']}', fontsize=12)")
        if spec.properties.get('ylabel'):
            code_lines.append(f"ax.set_ylabel('{spec.properties['ylabel']}', fontsize=12)")
        if spec.properties.get('xlim'):
            code_lines.append(f"ax.set_xlim{spec.properties['xlim']}")
        if spec.properties.get('ylim'):
            code_lines.append(f"ax.set_ylim{spec.properties['ylim']}")
        if spec.properties.get('grid'):
            code_lines.append("ax.grid(True, alpha=0.3)")
        
        code_lines.extend([
            "plt.tight_layout()",
            "plt.show()"
        ])
        
        return "\n".join(code_lines)
    
    def _generate_bar_chart_code(self, spec: DiagramSpec) -> str:
        """Generate code for bar chart"""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "fig, ax = plt.subplots(figsize=(8, 6))"
        ]
        
        x_positions = [element['x'] for element in spec.elements]
        heights = [element['height'] for element in spec.elements]
        colors = [element['color'] for element in spec.elements]
        
        code_lines.append(f"x_pos = {x_positions}")
        code_lines.append(f"heights = {heights}")
        code_lines.append(f"colors = {colors}")
        code_lines.append("bars = ax.bar(x_pos, heights, color=colors, alpha=0.7)")
        
        # Add properties
        if spec.properties.get('title'):
            code_lines.append(f"ax.set_title('{spec.properties['title']}', fontsize=12)")
        if spec.properties.get('xlabel'):
            code_lines.append(f"ax.set_xlabel('{spec.properties['xlabel']}', fontsize=12)")
        if spec.properties.get('ylabel'):
            code_lines.append(f"ax.set_ylabel('{spec.properties['ylabel']}', fontsize=12)")
        if spec.properties.get('xlim'):
            code_lines.append(f"ax.set_xlim{spec.properties['xlim']}")
        if spec.properties.get('ylim'):
            code_lines.append(f"ax.set_ylim{spec.properties['ylim']}")
        if spec.properties.get('grid'):
            code_lines.append("ax.grid(True, alpha=0.3)")
        
        code_lines.extend([
            "plt.tight_layout()",
            "plt.show()"
        ])
        
        return "\n".join(code_lines)
    
    def _generate_scatter_code(self, spec: DiagramSpec) -> str:
        """Generate code for scatter plot"""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "fig, ax = plt.subplots(figsize=(8, 6))"
        ]
        
        for element in spec.elements:
            x_data = element['x_data']
            y_data = element['y_data']
            color = element['color']
            marker = element['marker']
            
            code_lines.append(f"ax.scatter({x_data}, {y_data}, color='{color}', marker='{marker}', s=50)")
        
        # Add properties
        if spec.properties.get('title'):
            code_lines.append(f"ax.set_title('{spec.properties['title']}', fontsize=12)")
        if spec.properties.get('xlabel'):
            code_lines.append(f"ax.set_xlabel('{spec.properties['xlabel']}', fontsize=12)")
        if spec.properties.get('ylabel'):
            code_lines.append(f"ax.set_ylabel('{spec.properties['ylabel']}', fontsize=12)")
        if spec.properties.get('xlim'):
            code_lines.append(f"ax.set_xlim{spec.properties['xlim']}")
        if spec.properties.get('ylim'):
            code_lines.append(f"ax.set_ylim{spec.properties['ylim']}")
        if spec.properties.get('grid'):
            code_lines.append("ax.grid(True, alpha=0.3)")
        
        code_lines.extend([
            "plt.tight_layout()",
            "plt.show()"
        ])
        
        return "\n".join(code_lines)
    
    def _generate_pie_chart_code(self, spec: DiagramSpec) -> str:
        """Generate code for pie chart"""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "",
            "fig, ax = plt.subplots(figsize=(8, 6))"
        ]
        
        values = [element['value'] for element in spec.elements]
        labels = [element['label'] for element in spec.elements]
        colors = [element['color'] for element in spec.elements]
        
        code_lines.append(f"values = {values}")
        code_lines.append(f"labels = {labels}")
        code_lines.append(f"colors = {colors}")
        code_lines.append("ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)")
        
        # Add properties
        if spec.properties.get('title'):
            code_lines.append(f"ax.set_title('{spec.properties['title']}', fontsize=12)")
        
        code_lines.extend([
            "ax.axis('equal')",
            "plt.tight_layout()",
            "plt.show()"
        ])
        
        return "\n".join(code_lines)

class DiagramSolver:
    """Verifies and corrects Python programs (SOLVE role)"""
    
    def __init__(self):
        self.verifier = SemanticDiagramVerifier()
        self.correction_attempts = 0
        self.max_correction_attempts = 3
    
    def solve_problem(self, code: str, target_spec: DiagramSpec) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Attempt to solve the diagram generation problem"""
        # First, try to verify the original code
        success, score, details = self.verifier.verify_program(code, target_spec)
        
        if success:
            return True, code, score, details
        
        # If not successful, try to correct the code
        corrected_code = self._attempt_correction(code, target_spec)
        if corrected_code:
            corrected_success, corrected_score, corrected_details = self.verifier.verify_program(corrected_code, target_spec)
            if corrected_success:
                return True, corrected_code, corrected_score, corrected_details
        
        return False, code, score, details
    
    def _attempt_correction(self, code: str, target_spec: DiagramSpec) -> Optional[str]:
        """Attempt to correct the code to match the target specification"""
        if self.correction_attempts >= self.max_correction_attempts:
            return None
        
        self.correction_attempts += 1
        
        try:
            # Parse the code to understand its structure
            tree = ast.parse(code)
            
            # Simple corrections based on common issues
            corrected_code = self._apply_corrections(code, target_spec)
            
            return corrected_code
            
        except Exception as e:
            logger.warning(f"Error during code correction: {e}")
            return None
    
    def _apply_corrections(self, code: str, target_spec: DiagramSpec) -> str:
        """Apply specific corrections to the code"""
        corrected_lines = code.split('\n')
        
        # Ensure proper imports
        if 'import matplotlib.pyplot as plt' not in code:
            corrected_lines.insert(0, 'import matplotlib.pyplot as plt')
        if 'import numpy as np' not in code:
            corrected_lines.insert(1, 'import numpy as np')
        
        # Ensure proper figure creation
        if 'plt.subplots' not in code:
            # Find where to insert figure creation
            for i, line in enumerate(corrected_lines):
                if 'ax.plot' in line or 'ax.scatter' in line or 'ax.bar' in line:
                    corrected_lines.insert(i, 'fig, ax = plt.subplots(figsize=(8, 6))')
                    break
        
        # Ensure proper show() call
        if 'plt.show()' not in code:
            corrected_lines.append('plt.show()')
        
        return '\n'.join(corrected_lines)

class SelfPlayTrainer:
    """Main trainer that orchestrates the self-play loop"""
    
    def __init__(self, 
                 output_dir: str = "dac_azr_output",
                 population_size: int = 100,
                 max_generations: int = 50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.population_size = population_size
        self.max_generations = max_generations
        
        self.proposer = DiagramProposer()
        self.solver = DiagramSolver()
        self.verifier = SemanticDiagramVerifier()
        
        # Population of successful (diagram, code) pairs
        self.population: List[CodeProgram] = []
        
        # Statistics
        self.generation_stats = []
        self.best_scores = []
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the training session"""
        log_file = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def initialize_population(self):
        """Initialize the population with seed examples"""
        logger.info("Initializing population with seed examples...")
        
        # Generate initial seed examples
        for i in range(min(10, self.population_size)):
            try:
                # Generate a diagram concept
                diagram_spec = self.proposer.propose_diagram_concept()
                
                # Generate Python code for it
                code = self.proposer.generate_python_program(diagram_spec)
                
                # Verify the code works
                success, score, details = self.verifier.verify_program(code, diagram_spec)
                
                if success:
                    program = CodeProgram(
                        code=code,
                        diagram_spec=diagram_spec,
                        success_score=score,
                        complexity_score=diagram_spec.complexity_score,
                        generation_metadata={
                            'generation': 0,
                            'method': 'seed',
                            'corrections': 0
                        }
                    )
                    
                    self.population.append(program)
                    logger.info(f"Added seed example {i+1}: {diagram_spec.diagram_type} (score: {score:.3f})")
                
            except Exception as e:
                logger.warning(f"Error generating seed example {i+1}: {e}")
        
        logger.info(f"Population initialized with {len(self.population)} examples")
    
    def evolve_generation(self, generation: int):
        """Evolve to the next generation using self-play"""
        logger.info(f"Starting generation {generation}")
        
        new_programs = []
        
        # Generate new programs through self-play
        for i in range(self.population_size):
            try:
                # PROPOSE: Generate new diagram concept and code
                diagram_spec = self.proposer.propose_diagram_concept()
                code = self.proposer.generate_python_program(diagram_spec)
                
                # SOLVE: Verify and potentially correct the code
                success, final_code, score, details = self.solver.solve_problem(code, diagram_spec)
                
                if success:
                    program = CodeProgram(
                        code=final_code,
                        diagram_spec=diagram_spec,
                        success_score=score,
                        complexity_score=diagram_spec.complexity_score,
                        generation_metadata={
                            'generation': generation,
                            'method': 'self_play',
                            'corrections': self.solver.correction_attempts,
                            'original_score': details.get('overall_score', 0.0)
                        }
                    )
                    
                    new_programs.append(program)
                    logger.info(f"Generated program {i+1}: {diagram_spec.diagram_type} (score: {score:.3f})")
                
                # Reset correction attempts for next program
                self.solver.correction_attempts = 0
                
            except Exception as e:
                logger.warning(f"Error in generation {generation}, program {i+1}: {e}")
        
        # Update population
        if new_programs:
            # Keep best programs from both old and new
            all_programs = self.population + new_programs
            all_programs.sort(key=lambda x: x.success_score, reverse=True)
            
            # Keep top performers
            self.population = all_programs[:self.population_size]
            
            # Update statistics
            best_score = self.population[0].success_score if self.population else 0.0
            avg_score = np.mean([p.success_score for p in self.population]) if self.population else 0.0
            
            self.best_scores.append(best_score)
            
            generation_stat = {
                'generation': generation,
                'best_score': best_score,
                'avg_score': avg_score,
                'population_size': len(self.population),
                'new_programs': len(new_programs)
            }
            self.generation_stats.append(generation_stat)
            
            logger.info(f"Generation {generation} completed: Best={best_score:.3f}, Avg={avg_score:.3f}, Population={len(self.population)}")
        else:
            logger.warning(f"Generation {generation} failed to generate any new programs")
    
    def train(self):
        """Run the complete training process"""
        logger.info("Starting DAC-AZR training...")
        
        # Initialize population
        self.initialize_population()
        
        if not self.population:
            logger.error("Failed to initialize population. Exiting.")
            return
        
        # Training loop
        for generation in range(1, self.max_generations + 1):
            try:
                self.evolve_generation(generation)
                
                # Save progress periodically
                if generation % 10 == 0:
                    self.save_progress(generation)
                
                # Early stopping if no improvement
                if len(self.best_scores) >= 5:
                    recent_improvement = max(self.best_scores[-5:]) - min(self.best_scores[-5:])
                    if recent_improvement < 0.01:
                        logger.info(f"Early stopping at generation {generation} due to no improvement")
                        break
                
            except Exception as e:
                logger.error(f"Error in generation {generation}: {e}")
                continue
        
        # Final save
        self.save_final_results()
        logger.info("Training completed!")
    
    def save_progress(self, generation: int):
        """Save progress at a specific generation"""
        progress_file = self.output_dir / f"progress_gen_{generation}.pkl"
        
        progress_data = {
            'generation': generation,
            'population': self.population,
            'generation_stats': self.generation_stats,
            'best_scores': self.best_scores
        }
        
        with open(progress_file, 'wb') as f:
            pickle.dump(progress_data, f)
        
        logger.info(f"Progress saved for generation {generation}")
    
    def save_final_results(self):
        """Save final training results"""
        # Save final population
        final_population_file = self.output_dir / "final_population.pkl"
        with open(final_population_file, 'wb') as f:
            pickle.dump(self.population, f)
        
        # Save statistics
        stats_file = self.output_dir / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.generation_stats, f, indent=2)
        
        # Save best programs as Python files
        best_programs_dir = self.output_dir / "best_programs"
        best_programs_dir.mkdir(exist_ok=True)
        
        for i, program in enumerate(self.population[:10]):  # Top 10
            program_file = best_programs_dir / f"best_program_{i+1}.py"
            with open(program_file, 'w') as f:
                f.write(program.code)
            
            # Save diagram spec
            spec_file = best_programs_dir / f"best_program_{i+1}_spec.json"
            with open(spec_file, 'w') as f:
                json.dump(program.diagram_spec.to_dict(), f, indent=2)
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info(f"Final results saved to {self.output_dir}")
    
    def _generate_summary_report(self):
        """Generate a summary report of the training"""
        report_file = self.output_dir / "training_summary.md"
        
        with open(report_file, 'w') as f:
            f.write("# DAC-AZR Training Summary\n\n")
            f.write(f"**Training completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total generations:** {len(self.generation_stats)}\n")
            f.write(f"**Final population size:** {len(self.population)}\n\n")
            
            if self.best_scores:
                f.write(f"**Best score achieved:** {max(self.best_scores):.4f}\n")
                f.write(f"**Final best score:** {self.best_scores[-1]:.4f}\n\n")
            
            f.write("## Generation Statistics\n\n")
            f.write("| Generation | Best Score | Avg Score | Population | New Programs |\n")
            f.write("|------------|------------|-----------|------------|--------------|\n")
            
            for stat in self.generation_stats:
                f.write(f"| {stat['generation']} | {stat['best_score']:.4f} | {stat['avg_score']:.4f} | {stat['population_size']} | {stat['new_programs']} |\n")
            
            f.write("\n## Top Programs\n\n")
            for i, program in enumerate(self.population[:5]):
                f.write(f"### Program {i+1}\n")
                f.write(f"- **Type:** {program.diagram_spec.diagram_type}\n")
                f.write(f"- **Score:** {program.success_score:.4f}\n")
                f.write(f"- **Complexity:** {program.complexity_score:.4f}\n")
                f.write(f"- **Generation:** {program.generation_metadata['generation']}\n\n")
    
    def export_dataset(self, output_file: str = "dac_azr_dataset.jsonl"):
        """Export the generated dataset in JSONL format"""
        dataset_file = self.output_dir / output_file
        
        with open(dataset_file, 'w') as f:
            for program in self.population:
                dataset_entry = {
                    'diagram_spec': program.diagram_spec.to_dict(),
                    'python_code': program.code,
                    'success_score': program.success_score,
                    'complexity_score': program.complexity_score,
                    'metadata': program.generation_metadata
                }
                f.write(json.dumps(dataset_entry) + '\n')
        
        logger.info(f"Dataset exported to {dataset_file}")
        return dataset_file

def main():
    """Main function to run the DAC-AZR system"""
    print("Diagram-as-Code Absolute Zero Reasoner (DAC-AZR)")
    print("=" * 60)
    
    # Create trainer
    trainer = SelfPlayTrainer(
        output_dir="dac_azr_output",
        population_size=50,  # Smaller for demonstration
        max_generations=20   # Fewer generations for demonstration
    )
    
    try:
        # Run training
        trainer.train()
        
        # Export dataset
        dataset_file = trainer.export_dataset()
        print(f"\nTraining completed! Dataset exported to: {dataset_file}")
        
        # Show some statistics
        if trainer.population:
            print(f"\nFinal Statistics:")
            print(f"- Population size: {len(trainer.population)}")
            print(f"- Best score: {max(trainer.best_scores):.4f}")
            print(f"- Average score: {np.mean([p.success_score for p in trainer.population]):.4f}")
            
            # Show top programs
            print(f"\nTop 3 Programs:")
            for i, program in enumerate(trainer.population[:3]):
                print(f"{i+1}. {program.diagram_spec.diagram_type} - Score: {program.success_score:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 