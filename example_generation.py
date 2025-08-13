#!/usr/bin/env python3
"""
Example script showing how to generate specific types of diagrams
and their corresponding Python code using the DAC-AZR system.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diagram_as_code_azr import DiagramProposer, SemanticDiagramVerifier

def generate_line_plot_example():
    """Generate a line plot example"""
    print("=" * 50)
    print("Line Plot Generation Example")
    print("=" * 50)
    
    proposer = DiagramProposer()
    verifier = SemanticDiagramVerifier()
    
    # Generate a line plot concept
    diagram_spec = proposer.propose_diagram_concept()
    # Force it to be a line plot
    diagram_spec.diagram_type = 'line_plot'
    
    print(f"Generated diagram specification:")
    print(f"  Type: {diagram_spec.diagram_type}")
    print(f"  Elements: {len(diagram_spec.elements)}")
    print(f"  Complexity: {diagram_spec.complexity_score:.3f}")
    
    # Generate Python code
    code = proposer.generate_python_program(diagram_spec)
    
    print(f"\nGenerated Python code:")
    print("-" * 30)
    print(code)
    print("-" * 30)
    
    # Verify the code
    success, score, details = verifier.verify_program(code, diagram_spec)
    
    print(f"\nVerification results:")
    print(f"  Success: {'Yes' if success else 'No'}")
    print(f"  Score: {score:.3f}")
    print(f"  Details: {details}")
    
    return diagram_spec, code, success, score

def generate_bar_chart_example():
    """Generate a bar chart example"""
    print("\n" + "=" * 50)
    print("Bar Chart Generation Example")
    print("=" * 50)
    
    proposer = DiagramProposer()
    verifier = SemanticDiagramVerifier()
    
    # Generate a bar chart concept
    diagram_spec = proposer.propose_diagram_concept()
    # Force it to be a bar chart
    diagram_spec.diagram_type = 'bar_chart'
    
    print(f"Generated diagram specification:")
    print(f"  Type: {diagram_spec.diagram_type}")
    print(f"  Elements: {len(diagram_spec.elements)}")
    print(f"  Complexity: {diagram_spec.complexity_score:.3f}")
    
    # Generate Python code
    code = proposer.generate_python_program(diagram_spec)
    
    print(f"\nGenerated Python code:")
    print("-" * 30)
    print(code)
    print("-" * 30)
    
    # Verify the code
    success, score, details = verifier.verify_program(code, diagram_spec)
    
    print(f"\nVerification results:")
    print(f"  Success: {'Yes' if success else 'No'}")
    print(f"  Score: {score:.3f}")
    print(f"  Details: {details}")
    
    return diagram_spec, code, success, score

def generate_scatter_plot_example():
    """Generate a scatter plot example"""
    print("\n" + "=" * 50)
    print("Scatter Plot Generation Example")
    print("=" * 50)
    
    proposer = DiagramProposer()
    verifier = SemanticDiagramVerifier()
    
    # Generate a scatter plot concept
    diagram_spec = proposer.propose_diagram_concept()
    # Force it to be a scatter plot
    diagram_spec.diagram_type = 'scatter'
    
    print(f"Generated diagram specification:")
    print(f"  Type: {diagram_spec.diagram_type}")
    print(f"  Elements: {len(diagram_spec.elements)}")
    print(f"  Complexity: {diagram_spec.complexity_score:.3f}")
    
    # Generate Python code
    code = proposer.generate_python_program(diagram_spec)
    
    print(f"\nGenerated Python code:")
    print("-" * 30)
    print(code)
    print("-" * 30)
    
    # Verify the code
    success, score, details = verifier.verify_program(code, diagram_spec)
    
    print(f"\nVerification results:")
    print(f"  Success: {'Yes' if success else 'No'}")
    print(f"  Score: {score:.3f}")
    print(f"  Details: {details}")
    
    return diagram_spec, code, success, score

def generate_pie_chart_example():
    """Generate a pie chart example"""
    print("\n" + "=" * 50)
    print("Pie Chart Generation Example")
    print("=" * 50)
    
    proposer = DiagramProposer()
    verifier = SemanticDiagramVerifier()
    
    # Generate a pie chart concept
    diagram_spec = proposer.propose_diagram_concept()
    # Force it to be a pie chart
    diagram_spec.diagram_type = 'pie_chart'
    
    print(f"Generated diagram specification:")
    print(f"  Type: {diagram_spec.diagram_type}")
    print(f"  Elements: {len(diagram_spec.elements)}")
    print(f"  Complexity: {diagram_spec.complexity_score:.3f}")
    
    # Generate Python code
    code = proposer.generate_python_program(diagram_spec)
    
    print(f"\nGenerated Python code:")
    print("-" * 30)
    print(code)
    print("-" * 30)
    
    print(f"\nNote: Pie chart verification may require matplotlib backend setup")
    print(f"Generated code should create a pie chart with {len(diagram_spec.elements)} slices")
    
    return diagram_spec, code

def demonstrate_custom_diagram():
    """Demonstrate creating a custom diagram specification"""
    print("\n" + "=" * 50)
    print("Custom Diagram Specification Example")
    print("=" * 50)
    
    from diagram_as_code_azr import DiagramSpec, DiagramProposer
    
    # Create a custom diagram specification
    custom_spec = DiagramSpec(
        diagram_type="line_plot",
        elements=[
            {
                "type": "line",
                "x_data": [0, 1, 2, 3, 4, 5],
                "y_data": [0, 1, 4, 9, 16, 25],
                "color": "r",
                "linestyle": "-",
                "marker": "o"
            },
            {
                "type": "line",
                "x_data": [0, 1, 2, 3, 4, 5],
                "y_data": [0, 2, 4, 6, 8, 10],
                "color": "b",
                "linestyle": "--",
                "marker": "s"
            }
        ],
        properties={
            "xlabel": "X Values",
            "ylabel": "Y Values",
            "title": "Custom Plot: Quadratic vs Linear",
            "xlim": (0, 5),
            "ylim": (0, 25),
            "grid": True
        },
        complexity_score=0.6
    )
    
    print("Custom diagram specification created:")
    print(f"  Type: {custom_spec.diagram_type}")
    print(f"  Elements: {len(custom_spec.elements)}")
    print(f"  Complexity: {custom_spec.complexity_score:.3f}")
    
    # Generate code for the custom specification
    proposer = DiagramProposer()
    code = proposer.generate_python_program(custom_spec)
    
    print(f"\nGenerated Python code for custom specification:")
    print("-" * 30)
    print(code)
    print("-" * 30)
    
    return custom_spec, code

def main():
    """Run all examples"""
    print("DAC-AZR Example Generation")
    print("=" * 60)
    
    try:
        # Generate examples for different diagram types
        line_spec, line_code, line_success, line_score = generate_line_plot_example()
        bar_spec, bar_code, bar_success, bar_score = generate_bar_chart_example()
        scatter_spec, scatter_code, scatter_success, scatter_score = generate_scatter_plot_example()
        pie_spec, pie_code = generate_pie_chart_example()
        
        # Demonstrate custom diagram creation
        custom_spec, custom_code = demonstrate_custom_diagram()
        
        # Summary
        print("\n" + "=" * 60)
        print("Example Generation Summary")
        print("=" * 60)
        print(f"Line Plot: {'✓' if line_success else '✗'} (Score: {line_score:.3f})")
        print(f"Bar Chart: {'✓' if bar_success else '✗'} (Score: {bar_score:.3f})")
        print(f"Scatter Plot: {'✓' if scatter_success else '✗'} (Score: {scatter_score:.3f})")
        print(f"Pie Chart: Generated (verification skipped)")
        print(f"Custom Diagram: Generated")
        
        print(f"\nAll examples generated successfully!")
        print(f"You can now run the generated code to see the actual plots.")
        
    except Exception as e:
        print(f"Example generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 