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

Both tasks involve predicting labels based on input data, classification provides a single label for the entire image, whereas segmentation provides a detailed label for every pixel in the image. This makes segmentation a more complex and resource-intensive task compared to classification.

9. ***What is data and dataset difference?***

_Data_ is individual pieces of information, like numbers, words, or images. It's the raw material.   

_Dataset_ is a collection of related data organized in a specific way. It's like a container for data, often used for analysis or machine learning.

10. ***What is the difference between supervised and unsupervised learning in terms of dataset?***

i. Supervised Learning: Requires a labeled dataset, where each data instance is paired with its corresponding output label.

ii. Unsupervised Learning: Works with an unlabeled dataset, meaning there are no explicit output labels provided for the data instances.



## Data Preprocessing

### Extracting Masks

- ***What is color space ?***

Color space is a specific organization or model for representing colors. It allows for the consistent interpretation and manipulation of color information by algorithms and neural networks. Understanding and using different color spaces can significantly impact the performance and accuracy of AI models dealing with visual data.

- ***What RGB stands for ?***

RGB stands for Red, Green, and Blue, a color model where colors are represented as combinations of these three primary colors.

- ***In Python, can we transform from one color space to another?***

Yes, in Python, we can transform from one color space to another using various libraries and functions. One popular library for working with colors and color spaces is OpenCV. OpenCV provides functions to convert images between different color spaces, such as RGB to grayscale, RGB to HSV, RGB to LAB, etc.

- ***What is the popular library for image processing?***

i.  OpenCV: Best for a comprehensive range of image processing and computer vision tasks.

ii.  Pillow: Ideal for basic image manipulation.

iii.  scikit-image: Great for scientific and advanced image processing.

iv.  TensorFlow/Keras: Useful for integrating image processing with deep learning models.

v. PyTorch: Provides robust tools for image preprocessing in deep learning contexts.

### Converting into Tensor

- ***Explain Computational Graph.***

A computational graph is a graphical representation of a mathematical expression or a computational process. It visualizes the flow of data through operations, making it easier to understand and optimize the process, especially in the context of deep learning and machine learning.

- ***What is Tensor?***

A tensor is a mathematical object that generalizes scalars, vectors, and matrices to higher dimensions. In the context of computing and machine learning, particularly with frameworks like TensorFlow and PyTorch, tensors are multi-dimensional arrays used to store and manipulate data.

- ***What is one hot encoding?***

One-hot encoding is a technique to convert categorical data into a binary format where each category is represented by a unique binary vector. This method is widely used in machine learning to handle categorical variables and ensure that models can effectively interpret and process categorical information.

- ***What is CUDA programming?***

CUDA programming is a parallel computing platform and API model developed by NVIDIA that allows developers to use NVIDIA GPUs for general-purpose processing. It enables faster computation by offloading compute-intensive tasks from the CPU to the GPU.


## Design Segmentation Model

- ***What is the difference between CNN and Fully CNN (FCNN) ?***

CNNs are designed for fixed-size outputs, FCNNs are designed to handle inputs and outputs of variable sizes, making them suitable for tasks requiring spatial outputs.

- ***What are the different layers on CNN ?***

CNN layers:

i.  Convolutional layer: Applies filters to extract features.  

ii. Pooling layer: Downsamples feature maps to reduce computational cost.   

iii.Fully connected layer: Classifies input based on learned features.

iv. Activation layer: Introduces non-linearity (e.g., ReLU).   

v. Normalization layer: Improves training stability (e.g., Batch normalization).

- ***What is activation function ? Why is softmax usually used in the last layer?***

_Activation Function:_ Introduces non-linearity and enables learning complex patterns.

_Softmax in the Last Layer:_ Used to convert logits into a probability distribution for multi-class classification problems, making it easier to interpret the model’s predictions.






