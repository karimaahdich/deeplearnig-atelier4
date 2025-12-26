# Deep Learning Lab: Autoencoders, Variational Autoencoders, and GANs

## Objective

The primary goal of this lab is to become familiar with the PyTorch library by building and training deep neural network architectures for Autoencoders (AE), Variational Autoencoders (VAE), and Generative Adversarial Networks (GANs). We use the MNIST dataset for AE and VAE, and the Abstract Art Gallery dataset for GANs.

This project was implemented using Google Colab/Kaggle for training and GitHub for version control.

## Datasets

- **MNIST Dataset:** Handwritten digits (0-9), used for AE and VAE. Source: Kaggle MNIST Dataset.
- **Abstract Art Gallery Dataset:** Abstract images for GAN training. Source: Kaggle Abstract Art Gallery.

## Part 1: Autoencoder (AE) and Variational Autoencoder (VAE) on MNIST

### 1. Autoencoder (AE) Architecture and Training

- **Architecture:**
  - Encoder: Fully connected layers (784 → 128 → 64 → 32) to compress input to latent space.
  - Decoder: Symmetric to encoder (32 → 64 → 128 → 784) to reconstruct the input.
  - Activation: ReLU for hidden layers, Sigmoid for output.
- **Hyperparameters:**
  - Latent dimension: 32
  - Batch size: 128
  - Learning rate: 0.001
  - Optimizer: Adam
  - Loss function: Mean Squared Error (MSE)
  - Epochs: 50
- **Training:** Trained on MNIST training set. Loss decreased steadily, indicating good convergence.

### 2. Variational Autoencoder (VAE) Architecture and Training

- **Architecture:**
  - Encoder: Similar to AE but outputs mean and log-variance for latent space (784 → 128 → 64 → 32*2).
  - Decoder: Samples from latent distribution and reconstructs (32 → 64 → 128 → 784).
  - Activation: ReLU for hidden layers, Sigmoid for output.
  - Reparameterization trick used for sampling.
- **Hyperparameters:**
  - Latent dimension: 32
  - Batch size: 128
  - Learning rate: 0.001
  - Optimizer: Adam
  - Loss function: Reconstruction loss (MSE) + KL Divergence
  - Epochs: 50
- **Training:** Trained on MNIST. Total loss includes both reconstruction and regularization terms.

### 3. Evaluation

- **Loss Comparison:** AE loss decreases rapidly and plateaus lower than VAE.
- **KL Divergence (VAE):** Increases initially and stabilizes, ensuring latent space approximates standard normal distribution.
- **Conclusion:** AE achieves better reconstruction (lower loss) but lacks generative capability. VAE trades some reconstruction quality for a structured latent space, enabling generation of new samples.

### 4. Latent Space Visualization

- **VAE Latent Space (t-SNE):** Shows clear clustering of digits (0-9), demonstrating learned representations.
- **AE Latent Space:** Typically less structured than VAE.

## Part 2: Generative Adversarial Networks (GANs) on Abstract Art Gallery

### 1. GAN Architecture and Training

- **Generator:**
  - Input: Noise vector (latent dim=100)
  - Architecture: Fully connected or convolutional layers to upsample to image size (100 → 256 → 512 → image channels)
  - Activation: ReLU, Tanh for output
- **Discriminator:**
  - Input: Image (real or generated)
  - Architecture: Convolutional layers to classify real/fake (downsample to 1 output)
  - Activation: LeakyReLU, Sigmoid output
- **Loss Function:** Binary Cross-Entropy (adversarial loss for both G and D)
- **Initialization:** Weights initialized normally (mean=0, std=0.02)
- **GPU Setting:** CUDA used if available
- **Data Loader:** Batch size 64, transformations (resize, normalize)
- **Optimizers:** Adam for both G and D (lr=0.0002, betas=(0.5, 0.999))
- **Training:** Alternating updates for D and G over 100 epochs on Abstract Art images.

### 2. Evaluation

- **Loss Plots:** Generator loss decreases as it fools discriminator; Discriminator loss stabilizes.
- **KL Divergence:** Not directly applicable to vanilla GAN.
- **Conclusion:** GAN learns to generate abstract images. Training is unstable but produces diverse outputs with proper hyperparameters.

### 3. Generated Data

- Generated images show colorful, noisy patterns similar to the dataset, though with some artifacts.
- Quality improves with more epochs/training data.

## Synthesis

Through this lab, I learned:

- How to implement AE, VAE, and GAN from scratch in PyTorch, including custom architectures, loss functions, and training loops.
- Differences between AE (deterministic reconstruction) and VAE (probabilistic, generative latent space).
- GAN training dynamics: minimax game between generator and discriminator, handling instability.
- Evaluation techniques: Loss curves, latent space visualization (t-SNE), qualitative generation assessment.
- Practical skills: Hyperparameter tuning, dataset loading, GPU acceleration, debugging deep learning models.

This reinforces the foundations of generative AI and prepares for advanced topics like conditional GANs or diffusion models.
