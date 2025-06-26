# DEEP-LEARNING-PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: HARSHAD GANESH MORE

*INTERN ID*:CT04DG2387

*DOMAIN NAME*: DATA SCIENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

DESCRIPTION:
As part of my internship, I worked on a deep learning-based Natural Language Processing (NLP) project focused on text classification using PyTorch. The primary goal of this task was to develop a simple, yet effective machine learning model that could automatically categorize pieces of text into predefined classes. For this purpose, I used the AG_NEWS dataset, which is a popular benchmark for news classification. It contains thousands of news articles labeled into four categories: World, Sports, Business, and Sci/Tech.
The project was implemented using PyTorch and TorchText, two widely-used libraries for deep learning and text processing. The pipeline involved several stages including data loading, text preprocessing, model building, training, and evaluation.
To begin, the AG_NEWS dataset was loaded using TorchText’s built-in dataset loader. Each news entry consists of a label and a piece of text. For preprocessing, I used a basic English tokenizer and converted the text into numerical representations using a vocabulary built from the training dataset. This was necessary because neural networks work with numerical data, not raw text.
The neural network model I built was a simple feed-forward text classifier with an embedding layer. The embedding layer helps convert sparse word indices into dense vectors, capturing some semantic meaning. The output of the embedding was passed through a linear layer to generate predictions for each class. Since this was a classification task, the Cross Entropy Loss function was used to train the model, and the SGD (Stochastic Gradient Descent) optimizer was applied to update weights.
Training was done in multiple epochs, and for each epoch, the model's accuracy and loss were tracked. The model's performance improved gradually over epochs as it learned patterns from the training data. The results were visualized using a line graph showing how accuracy increased and loss decreased over time.
To make the project reusable and easy to test, I saved the trained model and generated a plot of the accuracy and loss values across epochs. The trained model was stored in a separate folder, and the graph was saved as an image in the outputs folder.
This project helped me strengthen my understanding of how natural language processing and deep learning can be combined to perform real-world tasks like text classification. It also taught me how to use key tools and libraries like PyTorch, TorchText, and Matplotlib, and how to structure a machine learning project from end to end.

#Output:
![Image](https://github.com/user-attachments/assets/0593e97e-2fd8-4ff7-a639-765c3fb87112)
