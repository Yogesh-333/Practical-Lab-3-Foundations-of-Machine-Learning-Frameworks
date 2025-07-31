# Practical Lab 3 — Dogs vs Cats: Deep Learning with Transfer Learning & Fine-Tuning
  
Author: Yogesh Kumar Gopal  
Course: Foundations of Machine Learning Frameworks (CSCN8010)

This repository contains a deep learning project focused on solving the binary image classification problem of distinguishing between cats and dogs. The project explores two distinct approaches: building a custom Convolutional Neural Network (CNN) from scratch and leveraging transfer learning by fine-tuning a pre-trained VGG16 model.

## 1. Introduction
The primary goal of this project is to demonstrate and compare the effectiveness of different deep learning strategies for image classification. We delve into data preparation, model building, training, evaluation, and analysis of model performance, highlighting the strengths and weaknesses of each approach.

## 2. Dataset
The project utilizes the widely known Kaggle Dogs vs. Cats dataset, which consists of a large collection of images of dogs and cats. For this assignment, a subset of 5,000 images was used for training and validation to manage computational resources while still providing a robust dataset for experimentation.

## 3. Methodology
Custom Convolutional Neural Network (CNN)
A custom CNN was designed and trained from the ground up. This involved defining the network architecture, including convolutional layers, pooling layers, and dense layers, tailored for the specific task.

### Transfer Learning with VGG16
The project extensively uses transfer learning by fine-tuning the pre-trained VGG16 model. VGG16, trained on the vast ImageNet dataset, provides a powerful set of learned features. By freezing the initial layers and fine-tuning the later layers (or adding new classification layers), we adapt this powerful model to our specific Dogs vs. Cats dataset.

### Data Augmentation and Callbacks
To improve model generalization and prevent overfitting, various techniques were employed:

### Data Augmentation: Techniques like rotation, shifting, zooming, and flipping were applied to the training images to create more diverse training examples.

### Callbacks: Early stopping and model checkpointing were used during training to monitor performance, save the best model, and prevent unnecessary training epochs.

## 4. Discussion and Conclusions
Our analysis revealed significant insights into the efficacy of both approaches:

### Model Performance: The fine-tuned VGG16 model generally outperformed the custom CNN. This is largely attributed to VGG16's ability to leverage pre-learned, robust features from a much larger and more diverse dataset (ImageNet), which are highly transferable to similar image classification tasks. The custom CNN, while effective, required more data and careful tuning to reach comparable performance levels.

### Impact of Transfer Learning: Transfer learning proved to be a highly effective strategy, significantly accelerating the learning process. By starting with pre-trained weights, the model converged faster and achieved higher accuracy with a relatively smaller dataset compared to training a complex CNN from scratch. This demonstrates the power of utilizing knowledge gained from large-scale datasets.

### Overfitting/Underfitting: Both models were susceptible to overfitting, especially the custom CNN. Data augmentation played a crucial role in mitigating this by increasing the diversity of the training data. Callbacks, particularly early stopping, were vital in preventing the models from training too long and memorizing the training data, ensuring better generalization to unseen images.

### Common Failure Cases: Both models occasionally struggled with images that had ambiguous features, unusual lighting conditions, or contained multiple animals where the primary subject was unclear. Images with heavy blurring or highly artistic renditions of cats or dogs also posed challenges.

Next Steps for Performance Improvement: To further boost performance, future work could include experimenting with other state-of-the-art pre-trained models (e.g., ResNet, Inception), implementing more advanced data augmentation techniques, or exploring ensemble methods by combining predictions from multiple models. Collecting a larger and more diverse dataset would also invariably lead to better generalization.

## 5. Takeaways
Transfer learning is a powerful technique that typically delivers strong performance gains, even with relatively small datasets, by leveraging knowledge from pre-trained models.

The choice of model (custom CNN vs. transfer learning) depends on several factors, including the size and nature of the dataset, available computational resources, and the specific requirement for maximum accuracy. For many real-world image classification tasks with limited data, transfer learning often provides a more efficient and effective solution.

## 6. Setup and Usage
###  Requirements

- Python 3.12+
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

Dataset:
Download from https://www.kaggle.com/datasets/biaiscience/dogs-vs-cats
