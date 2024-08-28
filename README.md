
## Project Overview

This project implements a Generative Adversarial Network (GAN) for generating cross-stitch patterns. The GAN is designed to generate images of cross-stitch designs by training on a dataset of existing patterns. The project leverages a Deep Convolutional GAN (DCGAN) architecture, which is a popular choice for generating realistic images.

## Project Structure

- **DCGAN.ipynb**: The main Jupyter Notebook that contains the entire pipeline for training the GAN model. It includes data loading, model initialization, training loops, and visualization of generated images.

- **data_collector.py**: A helper script to track and visualize the loss and accuracy of the Generator and Discriminator during training. This script also manages saving and loading of training progress data.

- **models.py**: This script defines the neural network architectures for the Generator, Discriminator, and the combined GAN model.

- **data/**: A folder that should contain the dataset used for training. The dataset should be organized in subfolders, where each subfolder represents a class of cross-stitch patterns.

- **Images/**: This directory is where the generated images during training will be saved.

- **STATE_DICTS/**: Directory to save and load the state dictionaries (weights) of the Generator and Discriminator models.

- **LOSS_ACCURACY/**: Directory where loss and accuracy values are saved during training for later visualization.

## Requirements

To run this project, you will need the following packages:

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- seaborn
- numpy
- tqdm
- pickle

You can install the required packages using the following command:

```bash
pip install torch torchvision matplotlib seaborn tqdm numpy
```

## Usage

### 1. Setting Up the Dataset

Place your dataset in the `data/` directory. The dataset should be organized into subfolders where each subfolder represents a different class of cross-stitch patterns.

### 2. Training the GAN

To train the GAN, simply run the `DCGAN.ipynb` notebook. You can control various aspects of the training process by modifying the parameters within the notebook, such as:

- `INIT`: Set this to `True` to start training from scratch, or `False` to resume training from saved weights.
- `TRAIN_MINUTES`: Duration of the training session in minutes.
- `EPOCH_MINUTES`: How often (in minutes) to save and visualize the generated images and training statistics.

### 3. Visualizing the Results

The `DataCollector` class in `data_collector.py` helps track and visualize the performance of the GAN during training. Losses and accuracies for both the Generator and Discriminator are plotted at regular intervals. Additionally, generated images are saved and displayed to monitor the progress of the Generator.

### 4. Saving and Loading the Model

The notebook allows saving the state of the models to the `STATE_DICTS/` directory. You can later reload these weights to resume training or to generate new images without retraining.

### 5. Generating Images

After training, you can generate new cross-stitch patterns using the trained Generator. The `DCGAN.ipynb` notebook provides a section at the end to load the trained Generator and create new images.

### 6. Customizing the Model

You can customize the model by modifying the `Generator` and `Discriminator` classes in the `models.py` file. The notebook is set up to easily integrate these changes for experimentation.


---![Generated_Sneakers](https://github.com/user-attachments/assets/1c02f48b-da2f-4a68-b804-f586cbe2f8cd)
