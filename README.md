# Machine Learning Project: Wine Quality Classification

This repository contains a series of Python notebooks implementing various machine learning algorithms to classify the quality of wine using the Wine Quality Dataset from the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/186/wine+quality).

## Project Description

The Wine Quality Dataset consists of 11 features and a quality score (ranging from 0 to 10) for each wine sample.  For the purpose of this project, the quality scores have been grouped into five categories:

*   **Very Bad:** Score 0-2
*   **Bad:** Score 2-4
*   **Alright:** Score 4-6
*   **Good:** Score 6-8
*   **Excellent:** Score 8-10

This project explores and implements several classification techniques to predict the wine quality category based on the given features.

## Notebooks

This repository contains four Python notebooks, each focusing on different machine learning algorithms:

1.  **MLAssignment1.ipynb:**
    *   Principal Component Analysis (PCA)
    *   Least Squares Regression (including Gradient Descent and Normal Equation)

2.  **MLAssignment2.ipynb:**
    *   Logistic Regression (Softmax Regression for multi-class classification)
    *   K-Nearest Neighbors (KNN)

3.  **MLAssignment3.ipynb:**
    *   Naive Bayes Classifier
    *   Multilayer Perceptron (MLP) in PyTorch

4.  **MLAssignment4.ipynb:**
    *   Support Vector Machine (SVM)
    *   K-Means Clustering (Note: K-Means is technically a clustering algorithm, but is included here for exploration)

## Data

The dataset used in this project is available in the `Data` folder:

*   `winequality-red.csv`: The original Wine Quality Dataset (red wine).

## Dependencies

The following Python libraries are required to run the notebooks:

*   `numpy`
*   `matplotlib`
*   `pandas`
*   `scikit-learn`
*   `torch`
*   `torchmetrics`

You can install these dependencies using pip:

```bash
pip install numpy matplotlib pandas scikit-learn torch torchmetrics
```

## Usage

Each notebook contains detailed explanations and code for implementing the respective machine learning algorithms. Simply open the notebooks in Jupyter Notebook or Google Colab and follow the instructions within.

## Key Implementation Details

*   **Data Preprocessing:** All notebooks start with loading the data, handling class labels (grouping quality scores), and normalizing features.
*   **Classification:** The goal is to accurately classify wine samples into one of the five quality categories.
*   **Evaluation:** Model performance is evaluated using appropriate metrics such as accuracy and error rates.
*   **One-Hot Encoding:** Utilized for multi-class classification in Logistic Regression, MLP and KNN.
*   **Cross-Validation:** Employed in KNN to select the best K value.

## Algorithms Implemented

*   **Principal Component Analysis (PCA):** Reduces the dimensionality of the dataset while preserving essential information.
*   **Least Squares Regression:**  A linear approach to modeling the relationship between features and the target variable. Implemented using both Gradient Descent and the Normal Equation.
*   **Logistic Regression:**  A linear model for classification, adapted for multi-class problems using the Softmax function.
*   **K-Nearest Neighbors (KNN):**  A non-parametric method that classifies samples based on the majority class among its K nearest neighbors.
*   **Naive Bayes Classifier:**  A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.
*   **Multilayer Perceptron (MLP):**  A feedforward neural network implemented in PyTorch.
*   **Support Vector Machine (SVM):** A powerful classification model that seeks to find the optimal hyperplane to separate classes.
*   **K-Means Clustering:** Divides the dataset into K clusters based on feature similarity.
