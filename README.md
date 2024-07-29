# Freespace Segmentation with Fully Convolutional Neural Network (FCNN)

1. ***What is Machine Learning***

Machine learning (ML) is a branch of artificial intelligence (AI) that enables computers to learn from data and improve their performance on tasks without being explicitly programmed. It involves using algorithms to identify patterns and make predictions based on input data.

2. ***What is Unsupervised vs Supervised learning difference?***

_Supervised Learning:_ Uses labeled data to train models for tasks like classification and regression. The goal is to predict outputs from new inputs based on the learned patterns.

_Unsupervised Learning:_ Uses unlabeled data to find patterns and structures, such as grouping similar items or reducing data dimensions. The goal is to explore data and uncover hidden patterns.
 

3. ***What is Deep Learning?***

Deep Learning is a subset of machine learning that involves training neural networks with many layers (called deep neural networks) to recognize patterns, make decisions, and learn complex representations from large amounts of data. It is particularly effective in tasks such as image and speech recognition, natural language processing, and more, where it can automatically learn features and representations from raw data.

4. ***What is Neural Network (NN)?*** 

Supervised learning uses labeled data with input-output pairs for training, while unsupervised learning uses unlabeled data and focuses on finding patterns or representations within the data without explicit output labels.

5. ***What is Convolution Neural Network (CNN)? Please give 2 advantages over NN.***

A Convolutional Neural Network (CNN) is a type of neural network designed specifically for processing grid-like data, such as images. It uses convolutional layers to automatically and adaptively learn spatial hierarchies of features from the input data.

*Advantages over traditional Neural Networks (NN)*

+ _Spatial Hierarchies:_ CNNs can capture spatial hierarchies and patterns in data (e.g., edges, textures, shapes) due to their convolutional layers, which makes them highly effective for image and video analysis.

+ _Parameter Sharing:_ CNNs use shared weights across different parts of the input, which reduces the number of parameters compared to fully connected layers in traditional NNs. This leads to more efficient training and better generalization.

6. ***What is segmentation task in NN? Is it supervised or unsupervised?***

Segmentation in neural networks refers to the task of dividing an image or dataset into distinct regions or segments based on certain criteria. This is often used in image processing to identify and separate different objects or areas within an image.Segmentation tasks are typically supervised. They require labeled data where the correct segmentation (i.e., the ground truth) is provided. The neural network learns from this labeled data to predict the segmentation for new, unseen images. However, there are some unsupervised approaches for segmentation, particularly in cases where labeled data is not available, but these are less common and often more challenging.

7. ***What is classification task in NN? Is it supervised or unsupervised?***

In neural networks, the classification task involves assigning input data to one of several predefined categories or classes. The model learns to distinguish between different classes based on labeled training data. Classification is typically supervised. It requires labeled data where each training example comes with a corresponding class label. The neural network learns from this labeled data to make accurate predictions on new, unseen data.

8.  ***Compare segmentation and classification in NN.***




