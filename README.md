# Simple Fully-Connected GAN for MNIST (TensorFlow)

This  contains a simple, stable implementation of a Generative Adversarial Network (GAN) using fully-connected layers in TensorFlow. It is designed to generate handwritten digits from the MNIST dataset.

## Highlights
- **Stable Training**: Converted from a robust PyTorch implementation.
- **Efficient Convergence**: Uses a large batch size (2048) for better gradient estimates.
- **Numerical Stability**: Utilizes logits for discriminator loss (avoiding Sigmoid saturation) and Tanh for generator output.
- **Simple Architecture**: Fully-connected dense layers make it easy to understand and fast to train on modern hardware (CPU/GPU).

## Architecture

### Generator
- **Input**: Latent noise vector (Dim: 100)
- **Layer 1**: Dense (128 units) + LeakyReLU (α=0.2) + Dropout (0.5)
- **Output Layer**: Dense (784 units) + Tanh Activation
- **Reshape**: Flat 784-dim vector back to 28x28 grayscale image.

### Discriminator
- **Input**: Flattened image (Dim: 784)
- **Layer 1**: Dense (128 units) + LeakyReLU (α=0.2) + Dropout (0.5)
- **Output Layer**: Dense (1 unit) - Outputs **Logits**.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Training Configuration
- **Epochs**: 100
- **Batch Size**: 2048
- **Optimizer**: Adam (Learning Rate: 0.0002, Beta 1: 0.5)
- **Loss Function**: Binary Crossentropy (with Logits)

## Usage
1. Open the notebook `simple-gan-tensorflow.ipynb` in any Jupyter environment (Local, Colab, or Kaggle).
2. Ensure GPU acceleration is enabled for faster training (though CPU works fine for this simple model).
3. Run all cells sequentially.
4. Visualizations of generated images will be displayed every 10 epochs.

## Results
The model quickly learns to generate recognizable digits. Example outputs are shown in the notebook directly after the training loop completes.
