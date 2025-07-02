# DCGANs
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** using PyTorch to generate synthetic Pokémon images. The pipeline downloads and processes the Pokémon image dataset, defines the architecture for both the generator and discriminator networks, and trains them using adversarial training.

#### Features:

* **Data Preparation**: Automatically downloads and extracts the Pokémon dataset. Images are resized to 64x64, normalized to \[-1, 1], and loaded using PyTorch's `ImageFolder`.
* **Generator (`net_G`)**: Composed of multiple `ConvTranspose2D` layers with batch normalization and ReLU activations. It maps a random noise vector to a 3x64x64 RGB image.
* **Discriminator (`net_D`)**: A CNN that distinguishes between real and generated images. It uses `Conv2D` layers with batch normalization and LeakyReLU activations.
* **Training Loop**: Trains the generator and discriminator using `BCEWithLogitsLoss` and the Adam optimizer. The generator aims to fool the discriminator, while the discriminator learns to distinguish real from fake.
* **Visualization**: Displays a grid of generated Pokémon images after each epoch to monitor training progress.

#### Hyperparameters:

* Latent vector size: 100
* Batch size: 256
* Learning rate: 0.005
* Epochs: 20
* Optimizer: Adam with betas (0.5, 0.999)

#### Output:

The trained generator creates visually plausible Pokémon-style images from random noise inputs, with results improving over epochs.

