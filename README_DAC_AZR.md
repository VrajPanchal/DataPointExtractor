# Diagram-as-Code Absolute Zero Reasoner (DAC-AZR)

A comprehensive system for generating synthetic datasets of diagrams and Python programs using the "Absolute Zero" self-play paradigm. This system implements a diagram-as-code environment where AI models can learn to generate and verify Python code that produces specific visual diagrams.

## 🎯 Overview

DAC-AZR follows the Absolute Zero reasoning paradigm:
1. **PROPOSE**: Generate diagram concepts and Python programs
2. **SOLVE**: Verify and correct the generated programs
3. **Self-play loop**: Continuous improvement through iterative refinement

The system creates a synthetic dataset of (diagram specification, Python code) pairs that can be used for training AI models to understand the relationship between visual concepts and their programmatic representations.

## 🏗️ Architecture

### Core Components

#### 1. Diagram Specification (`DiagramSpec`)
- **Semantic representation** of diagrams (not just visual)
- Supports multiple diagram types: line plots, bar charts, scatter plots, pie charts, heatmaps
- Includes complexity scoring for curriculum learning
- Structured data format for easy comparison and generation

#### 2. Diagram Proposer (`DiagramProposer`)
- **PROPOSE role** in the Absolute Zero paradigm
- Generates diverse diagram concepts with varying complexity
- Creates Python code intended to produce the target diagrams
- Implements diversity rewards to encourage exploration

#### 3. Diagram Solver (`DiagramSolver`)
- **SOLVE role** in the Absolute Zero paradigm
- Verifies if generated code produces the intended diagram
- Attempts to correct flawed code automatically
- Provides feedback for the self-play loop

#### 4. Semantic Verifier (`SemanticDiagramVerifier`)
- **Verification mechanism** using semantic comparison
- Extracts structured information from generated plots
- Compares diagram specifications rather than raw images
- More robust than visual comparison methods

#### 5. Self-Play Trainer (`SelfPlayTrainer`)
- Orchestrates the complete self-play loop
- Manages population evolution and selection
- Implements curriculum learning through complexity progression
- Exports synthetic datasets for downstream use

## 🚀 Key Features

### Semantic Verification
- **Not just visual comparison**: Compares underlying data structures and properties
- **Robust evaluation**: Handles rendering differences gracefully
- **Structured feedback**: Provides detailed analysis of mismatches

### Self-Play Learning
- **Continuous improvement**: Models learn from their own generated examples
- **Diversity promotion**: Encourages exploration of different diagram types
- **Complexity progression**: Automatically increases difficulty as models improve

### Flexible Diagram Generation
- **Multiple chart types**: Line plots, bar charts, scatter plots, pie charts, heatmaps
- **Configurable parameters**: Customizable ranges for data, styling, and complexity
- **Realistic examples**: Generates diagrams that mimic real-world use cases

### Dataset Export
- **JSONL format**: Standard format for machine learning pipelines
- **Rich metadata**: Includes success scores, complexity metrics, and generation info
- **Quality filtering**: Only exports verified, high-quality examples

## 📦 Installation

### Prerequisites
- Python 3.8+
- matplotlib
- numpy
- opencv-python
- scikit-image
- scikit-learn

### Quick Install
```bash
pip install -r dac_azr_requirements.txt
```

### From Source
```bash
git clone <repository-url>
cd dac-azr
pip install -e .
```

## 🎮 Usage

### Basic Usage

```python
from diagram_as_code_azr import DiagramProposer, SemanticDiagramVerifier

# Initialize components
proposer = DiagramProposer()
verifier = SemanticDiagramVerifier()

# Generate a diagram concept
diagram_spec = proposer.propose_diagram_concept()

# Generate Python code for it
code = proposer.generate_python_program(diagram_spec)

# Verify the code works
success, score, details = verifier.verify_program(code, diagram_spec)
```

### Self-Play Training

```python
from diagram_as_code_azr import SelfPlayTrainer

# Create trainer
trainer = SelfPlayTrainer(
    output_dir="output",
    population_size=100,
    max_generations=50
)

# Run training
trainer.train()

# Export dataset
dataset_file = trainer.export_dataset("my_dataset.jsonl")
```

### Configuration

The system can be configured using YAML files:

```yaml
# dac_azr_config.yaml
training:
  population_size: 100
  max_generations: 50

diagram_generation:
  complexity_range: [0.1, 1.0]
  supported_types:
    - line_plot
    - bar_chart
    - scatter
    - pie_chart
```

## 🔧 Customization

### Adding New Diagram Types

```python
class CustomDiagramVerifier(DiagramVerifier):
    def _verify_custom_type(self, code: str, target_spec: DiagramSpec):
        # Implement custom verification logic
        pass

class CustomDiagramProposer(DiagramProposer):
    def _generate_custom_concept(self) -> DiagramSpec:
        # Implement custom diagram generation
        pass
```

### Custom Verification Metrics

```python
class CustomVerifier(SemanticDiagramVerifier):
    def _compare_elements(self, target_elements, generated_elements):
        # Implement custom comparison logic
        pass
```

## 📊 Output Format

### Dataset Structure

Each entry in the generated dataset follows this structure:

```json
{
  "diagram_spec": {
    "diagram_type": "line_plot",
    "elements": [
      {
        "type": "line",
        "x_data": [1, 2, 3, 4, 5],
        "y_data": [2, 4, 6, 8, 10],
        "color": "b",
        "linestyle": "-",
        "marker": "o"
      }
    ],
    "properties": {
      "xlabel": "X Axis",
      "ylabel": "Y Axis",
      "title": "Sample Plot",
      "xlim": [0, 6],
      "ylim": [0, 12],
      "grid": true
    },
    "complexity_score": 0.45
  },
  "python_code": "import matplotlib.pyplot as plt\n...",
  "success_score": 0.92,
  "complexity_score": 0.45,
  "metadata": {
    "generation": 15,
    "method": "self_play",
    "corrections": 1
  }
}
```

### Training Outputs

- **Progress files**: Pickle files with intermediate training states
- **Best programs**: Python files of the highest-scoring generated code
- **Statistics**: JSON files with training metrics and generation statistics
- **Summary report**: Markdown file with training overview and results

## 🧪 Examples

### Running the Demo

```bash
python demo_dac_azr.py
```

This will demonstrate:
- Basic diagram generation and verification
- Code correction capabilities
- Self-play training process
- Dataset export functionality
- Interactive diagram generation

### Command Line Training

```bash
python diagram_as_code_azr.py --config dac_azr_config.yaml
```

## 🔬 Advanced Features

### Curriculum Learning
The system automatically increases complexity as models improve:
- **Adaptive difficulty**: Adjusts based on success rates
- **Complexity rewards**: Encourages generation of challenging examples
- **Diversity promotion**: Prevents overfitting to simple patterns

### Code Correction
Automatic correction of flawed generated code:
- **Syntax fixing**: Corrects common Python syntax errors
- **Structure completion**: Adds missing imports and function calls
- **Style improvement**: Ensures proper matplotlib usage

### Quality Control
Multiple layers of quality assurance:
- **Execution verification**: Ensures code runs without errors
- **Output validation**: Confirms generated diagrams match specifications
- **Complexity scoring**: Balances difficulty and learnability

## 📈 Performance

### Training Efficiency
- **Population evolution**: Efficient selection and mutation strategies
- **Parallel execution**: Support for multi-process training
- **Early stopping**: Automatic termination when no improvement is detected

### Memory Management
- **Incremental updates**: Only stores high-quality examples
- **Garbage collection**: Automatic cleanup of failed attempts
- **Checkpointing**: Regular saves to prevent data loss

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make your changes
5. Add tests
6. Submit a pull request

### Testing
```bash
pytest tests/
pytest --cov=diagram_as_code_azr tests/
```

## 📚 Research Applications

### Machine Learning Training
- **Supervised learning**: Train models to generate code from diagram descriptions
- **Reinforcement learning**: Use the self-play loop for policy improvement
- **Multi-modal learning**: Bridge visual and textual representations

### Data Augmentation
- **Synthetic datasets**: Generate large amounts of training data
- **Diversity injection**: Ensure coverage of different diagram types
- **Quality control**: Maintain high standards for generated examples

### Educational Applications
- **Programming education**: Teach matplotlib and data visualization
- **Code generation**: Demonstrate best practices for plotting
- **Debugging practice**: Learn to identify and fix plotting errors

## 🔮 Future Work

### Planned Features
- **3D visualization support**: Extend to 3D plots and surfaces
- **Interactive plots**: Support for Plotly and Bokeh
- **Natural language input**: Generate diagrams from text descriptions
- **Multi-language support**: Extend beyond Python/matplotlib

### Research Directions
- **Neural code generation**: Integrate with transformer models
- **Adversarial training**: Improve robustness through adversarial examples
- **Transfer learning**: Apply learned patterns to new diagram types
- **Human-in-the-loop**: Incorporate human feedback for quality improvement

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the "Absolute Zero" reasoning paradigm
- Built on the foundation of matplotlib and scientific Python ecosystem
- Developed for advancing AI capabilities in code generation and verification

## 📞 Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact the development team

---

**Note**: This system is designed for research and educational purposes. Generated code should be reviewed before use in production environments. 