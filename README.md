# Multiple_digit_recognition

# Overview

This document outlines the `DigitModel`, a neural network designed for multiple digit recognition in an image using PyTorch. The model combines Convolutional Neural Networks  and Gated Recurrent Units to handle images with sequences of digits. I used Connectionist Temporal Classification loss.

# Model Architecture
- Convolutional Layers: Extract features from input images.
- Linear and Dropout Layers:Transform features for sequential processing.
- GRU Layer:Captures sequence dependencies in digit arrangements.
- Output Layer:Predicts digit sequences.

# Forward Pass
The model processes image features through convolutional, linear, dropout, and GRU layers. It generates predictions and computes CTC loss when labels are provided.

# Handling Variable Label Lengths
To manage variable label lengths, labels are padded to a fixed size. CTC loss is used for accurate loss computation, particularly important in sequence modeling.


# Data Preparation for Training
Training data is prepared using a custom approach with the MNIST dataset

- Combining MNIST Images:Randomly selected digit images from the MNIST dataset are horizontally combined .
- Variable Digit Combinations: The number of digits combined in each image varies, typically between two to five digits, to mimic validation and test data provided.
- Label Generation:Corresponding labels for these combined images are generated, capturing the sequence of digits in each new image.
- Data Augmentation:This method also acts as a form of data augmentation, increasing the variability and quantity of training data.
- Saving Data: The combined images and their labels are saved, ready to be used for training the model.

# RUN
In the terminal just run this command : python train.py

# Conclusion
The `DigitModel` presents a approach to digit sequence recognition. But the model takes time for training to predict accurately. Inplace of the GRU layer an lstm layer can also be introduced and evaluated if it performs better. Will work on that shortly!!!

# Train Images 

![11983](https://github.com/Tarakzai/Multiple_digit_recognition/assets/80420558/3df9a953-cbe8-48f4-8e61-279c08cddc6f)

![11985](https://github.com/Tarakzai/Multiple_digit_recognition/assets/80420558/7f3a8f78-a99f-4990-9003-826d6b490ac5)

![11984](https://github.com/Tarakzai/Multiple_digit_recognition/assets/80420558/6b1b9416-9cd2-4cb6-a7ca-a1332ac0599c)






