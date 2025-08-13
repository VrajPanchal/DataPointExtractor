#!/usr/bin/env python3
"""
Demonstration script for the Diagram-as-Code Absolute Zero Reasoner (DAC-AZR)

This script demonstrates the key features of the system:
1. Diagram concept generation
2. Python code generation
3. Verification and correction
4. Self-play training
5. Dataset export
"""

import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diagram_as_code_azr import (
    DiagramProposer, 
    DiagramSolver, 
    SemanticDiagramVerifier,
    SelfPlayTrainer,
    DiagramSpec,
    CodeProgram
)

def demonstrate_basic_functionality():
    """Demonstrate basic diagram generation and verification"""
    print("=" * 60)
    print("DAC-AZR Basic Functionality Demonstration")
    print("=" * 60)
    
    # Initialize components
    proposer = DiagramProposer()
    solver = DiagramSolver()
    verifier = SemanticDiagramVerifier()
    
    # Generate a diagram concept
    print("\n1. Generating a diagram concept...")
    diagram_spec = proposer.propose_diagram_concept()
    print(f"   Generated: {diagram_spec.diagram_type}")
    print(f"   Elements: {len(diagram_spec.elements)}")
    print(f"   Complexity: {diagram_spec.complexity_score:.3f}")
    
    # Generate Python code
    print("\n2. Generating Python code...")
    code = proposer.generate_python_program(diagram_spec)
    print("   Code generated successfully!")
    
    # Show the first few lines of code
    code_lines = code.split('\n')
    print("   First 5 lines:")
    for i, line in enumerate(code_lines[:5]):
        print(f"   {i+1:2d}: {line}")
    if len(code_lines) > 5:
        print(f"   ... and {len(code_lines) - 5} more lines")
    
    # Verify the code
    print("\n3. Verifying the generated code...")
    success, score, details = verifier.verify_program(code, diagram_spec)
    print(f"   Verification result: {'SUCCESS' if success else 'FAILED'}")
    print(f"   Score: {score:.3f}")
    print(f"   Details: {details}")
    
    return diagram_spec, code, success, score

def demonstrate_code_correction():
    """Demonstrate code correction capabilities"""
    print("\n" + "=" * 60)
    print("Code Correction Demonstration")
    print("=" * 60)
    
    # Create a diagram spec
    proposer = DiagramProposer()
    diagram_spec = proposer.propose_diagram_concept()
    
    # Create intentionally flawed code
    flawed_code = """
import matplotlib.pyplot as plt
import numpy as np

# Missing figure creation
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
# Missing labels and show()
"""
    
    print("1. Flawed code:")
    print(flawed_code)
    
    # Try to solve the problem
    solver = DiagramSolver()
    success, corrected_code, score, details = solver.solve_problem(flawed_code, diagram_spec)
    
    print(f"\n2. Correction result: {'SUCCESS' if success else 'FAILED'}")
    print(f"   Score: {score:.3f}")
    
    if corrected_code != flawed_code:
        print("\n3. Corrected code:")
        print(corrected_code)
    else:
        print("\n3. No corrections were made")

def demonstrate_self_play_training():
    """Demonstrate the self-play training process"""
    print("\n" + "=" * 60)
    print("Self-Play Training Demonstration")
    print("=" * 60)
    
    # Create a small trainer for demonstration
    trainer = SelfPlayTrainer(
        output_dir="demo_output",
        population_size=20,  # Small for demo
        max_generations=5    # Few generations for demo
    )
    
    print("1. Initializing population...")
    trainer.initialize_population()
    print(f"   Initial population size: {len(trainer.population)}")
    
    if trainer.population:
        print(f"   Best initial score: {trainer.population[0].success_score:.3f}")
        
        print("\n2. Running training for 5 generations...")
        for gen in range(1, 6):
            trainer.evolve_generation(gen)
            
            if trainer.population:
                best_score = trainer.population[0].success_score
                avg_score = sum(p.success_score for p in trainer.population) / len(trainer.population)
                print(f"   Generation {gen}: Best={best_score:.3f}, Avg={avg_score:.3f}")
            else:
                print(f"   Generation {gen}: No valid programs")
        
        print("\n3. Training completed!")
        print(f"   Final population size: {len(trainer.population)}")
        if trainer.population:
            print(f"   Best final score: {trainer.population[0].success_score:.3f}")
        
        # Show some examples
        print("\n4. Top 3 generated programs:")
        for i, program in enumerate(trainer.population[:3]):
            print(f"   {i+1}. {program.diagram_spec.diagram_type} - Score: {program.success_score:.3f}")
            print(f"      Complexity: {program.complexity_score:.3f}")
            print(f"      Generation: {program.generation_metadata['generation']}")
    
    else:
        print("   Failed to initialize population")

def demonstrate_dataset_export():
    """Demonstrate dataset export functionality"""
    print("\n" + "=" * 60)
    print("Dataset Export Demonstration")
    print("=" * 60)
    
    # Create a simple dataset
    proposer = DiagramProposer()
    programs = []
    
    print("1. Generating sample programs...")
    for i in range(5):
        diagram_spec = proposer.propose_diagram_concept()
        code = proposer.generate_python_program(diagram_spec)
        
        program = CodeProgram(
            code=code,
            diagram_spec=diagram_spec,
            success_score=0.8 + i * 0.05,  # Simulated scores
            complexity_score=diagram_spec.complexity_score,
            generation_metadata={
                'generation': 0,
                'method': 'demo',
                'corrections': 0
            }
        )
        programs.append(program)
        print(f"   Generated program {i+1}: {diagram_spec.diagram_type}")
    
    # Export to JSONL
    output_file = "demo_dataset.jsonl"
    print(f"\n2. Exporting dataset to {output_file}...")
    
    with open(output_file, 'w') as f:
        for program in programs:
            dataset_entry = {
                'diagram_spec': program.diagram_spec.to_dict(),
                'python_code': program.code,
                'success_score': program.success_score,
                'complexity_score': program.complexity_score,
                'metadata': program.generation_metadata
            }
            f.write(json.dumps(dataset_entry) + '\n')
    
    print(f"   Dataset exported successfully!")
    print(f"   File size: {os.path.getsize(output_file)} bytes")
    
    # Show sample entry
    print("\n3. Sample dataset entry:")
    sample_entry = {
        'diagram_spec': programs[0].diagram_spec.to_dict(),
        'python_code': programs[0].code[:100] + "...",  # Truncated for display
        'success_score': programs[0].success_score,
        'complexity_score': programs[0].complexity_score,
        'metadata': programs[0].generation_metadata
    }
    print(json.dumps(sample_entry, indent=2))

def demonstrate_interactive_generation():
    """Demonstrate interactive diagram generation"""
    print("\n" + "=" * 60)
    print("Interactive Diagram Generation")
    print("=" * 60)
    
    proposer = DiagramProposer()
    verifier = SemanticDiagramVerifier()
    
    # Generate multiple examples
    diagram_types = ['line_plot', 'bar_chart', 'scatter', 'pie_chart']
    
    for diagram_type in diagram_types:
        print(f"\nGenerating {diagram_type}...")
        
        # Generate concept
        diagram_spec = proposer.propose_diagram_concept()
        # Force specific type for demonstration
        diagram_spec.diagram_type = diagram_type
        
        # Generate code
        code = proposer.generate_python_program(diagram_spec)
        
        # Verify
        success, score, details = verifier.verify_program(code, diagram_spec)
        
        print(f"   Type: {diagram_spec.diagram_type}")
        print(f"   Elements: {len(diagram_spec.elements)}")
        print(f"   Success: {'Yes' if success else 'No'}")
        print(f"   Score: {score:.3f}")
        
        # Show a snippet of the code
        code_lines = code.split('\n')
        print(f"   Code snippet: {code_lines[0]}")

def main():
    """Main demonstration function"""
    print("Diagram-as-Code Absolute Zero Reasoner (DAC-AZR)")
    print("Demonstration Script")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_basic_functionality()
        demonstrate_code_correction()
        demonstrate_self_play_training()
        demonstrate_dataset_export()
        demonstrate_interactive_generation()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        
        # Cleanup demo files
        if os.path.exists("demo_dataset.jsonl"):
            os.remove("demo_dataset.jsonl")
        if os.path.exists("demo_output"):
            import shutil
            shutil.rmtree("demo_output")
        
        print("\nDemo files cleaned up.")
        
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 