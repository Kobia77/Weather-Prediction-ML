# Weather Prediction Machine Learning

## Overview
Kobia77 Weather Prediction AI is a machine learning project developed in my third year of college that predicts weather conditions using Decision Trees and SVM on a global weather dataset. The project includes data preprocessing, SMOTE oversampling, model training, evaluation, and visualizations.

## Project Structure
    Directory structure:
    ├── Decision Trees.py
    ├── GlobalWeatherRepository.csv
    └── SVM Algorithm.py

- **Decision Trees.py**: Implements a Decision Tree classifier for weather prediction. It handles data cleaning, label encoding, oversampling with SMOTE, training, evaluation (classification report and confusion matrix), and visualizations (feature importance and learning curves).
- **SVM Algorithm.py**: Implements an SVM classifier that includes data scaling, SMOTE-based oversampling, training, evaluation, and feature importance visualization.
- **GlobalWeatherRepository.csv**: Contains the global weather data used for model training and evaluation.

## Features
- Data preprocessing and cleaning
- Handling imbalanced data with SMOTE
- Implementation of Decision Trees and SVM for weather prediction
- Comprehensive evaluation using classification reports and confusion matrices
- Visualizations of feature importance and learning curves
- Interactive user input for real-time predictions

## Installation
1. **Clone the Repository** 
2. **(Optional) Set Up a Virtual Environment**
3. **Install Required Packages**
    - pandas
    - numpy
    - scikit-learn
    - imbalanced-learn
    - seaborn
    - matplotlib

## Usage
- Run the Decision Tree Model:
```  python "Decision Trees.py"```
- Run the SVM Model:
```  python "SVM Algorithm.py"```

Follow the on-screen prompts to enter weather parameters for real-time predictions.

## Results and Evaluation
- Decision Trees.py outputs a classification report, a confusion matrix (with heatmap visualization), a feature importance bar chart, and a learning curve.
- SVM Algorithm.py displays training time, a classification report, a confusion matrix, and a feature importance visualization.

## Future Improvements
- Explore additional machine learning algorithms and ensemble methods
- Enhance data preprocessing and feature engineering techniques
- Develop a graphical user interface (GUI) for better user experience
- Optimize model performance for improved prediction accuracy



