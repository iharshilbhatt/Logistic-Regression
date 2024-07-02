# Logistic Regression with Python

This project implements logistic regression using gradient descent from scratch in Python. The dataset used contains exam scores of students and whether they were admitted to a university.

## Table of Contents
- [Description](#description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Results](#results)

## Description
This project demonstrates logistic regression, a binary classification algorithm. It uses gradient descent to optimize the cost function and predict the probability of student admission based on exam scores.

## Dataset
The dataset (\`ex2data1.txt\`) contains three columns:
- \`X\`: Exam 1 score
- \`Y\`: Exam 2 score
- \`Z\`: Admission (1 if admitted, 0 if not admitted)

## Installation
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/your-username/your-repository.git
   \`\`\`
2. Navigate to the project directory:
   \`\`\`bash
   cd your-repository
   \`\`\`
3. Install the required dependencies:
   \`\`\`bash
   pip install numpy matplotlib pandas
   \`\`\`

## Usage
1. Load the dataset and preprocess it:
   \`\`\`python
   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd

   columnname = ['X', 'Y', 'Z']
   dataset = pd.read_csv("ex2data1.txt", names=columnname)

**Navigate to the project directory:**
```
cd your-repository
```
**Install the required dependencies**
```
pip install numpy matplotlib pandas
```

**Run the Python script:**
```
python logistic_regression.py
```

The script will:
<ul>Load and preprocess the dataset.</ul>
<ul>Visualize the data.</ul>
<ul>Train the logistic regression model using gradient descent.</ul>
<ul>Plot the cost function history.</ul>
<ul>Visualize the decision boundary.</ul>
<ul>Make predictions for a given test example.</ul>
<ul>Print the training accuracy.</ul>

## Functions

<ul>sigmoid(z): Computes the sigmoid function.</ul>
<ul>costFunction(theta, X, y): Computes the logistic regression cost function and gradient.</ul>
<uL>featureNormalization(X): Normalizes features in the dataset.</ul>
<ul>gradientDescent(X, y, theta, alpha, num_iters): Performs gradient descent to optimize the cost function.</ul>
<ul>classifierPredict(theta, X): Predicts the class label for a given input.</ul>

## Results

The trained model can predict the probability of student admission based on their exam scores. The decision boundary is visualized, and the accuracy of the model on the training set is reported.
