# Multiple_digit_recognition

##Overview

This document outlines the `DigitModel`, a neural network designed for digit recognition tasks using PyTorch. The model combines Convolutional Neural Networks  and Gated Recurrent Units to handle images with sequences of digits. I used Connectionist Temporal Classification loss .In my code I have not addressed the  challenge of duplicate digit recognition in sequences.

##Model Architecture
- Convolutional Layers: Extract features from input images.
- Linear and Dropout Layers:Transform features for sequential processing.
- GRU Layer:Captures sequence dependencies in digit arrangements.
- Output Layer:Predicts digit sequences.

##Forward Pass
The model processes image features through convolutional, linear, dropout, and GRU layers. It generates predictions and computes CTC loss when labels are provided.

##Handling Variable Label Lengths
To manage variable label lengths, labels are padded to a fixed size. CTC loss is used for accurate loss computation, particularly important in sequence modeling.


##Data Preparation for Training
Training data is prepared using a custom approach with the MNIST dataset

- Combining MNIST Images:Randomly selected digit images from the MNIST dataset are horizontally combined .
- Variable Digit Combinations: The number of digits combined in each image varies, typically between two to five digits, to mimic validation and test data provided.
- Label Generation:Corresponding labels for these combined images are generated, capturing the sequence of digits in each new image.
- Data Augmentation:This method also acts as a form of data augmentation, increasing the variability and quantity of training data.
- Saving Data: The combined images and their labels are saved, ready to be used for training the model.
![image](https://github.com/Tarakzai/Multiple_digit_recognition/assets/80420558/cdf7b7e2-61d3-4106-b4e3-68000a7e07a5)
