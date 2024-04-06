# Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets
import time

# Define the Streamlit app
def app():
    st.subheader('Supervised Learning, Classification, and KNN with Diabetes Dataset')
    text = """**Supervised Learning:**
    \nSupervised learning is a branch of machine learning where algorithms learn from labeled data. 
    This data consists of input features (X) and corresponding outputs or labels (y). The algorithm learns a 
    mapping function from the input features to the outputs, allowing it to predict the labels for 
    unseen data points.
    \n**Classification:**
    Classification is a specific task within supervised learning where the labels belong to discrete 
    categories. The goal is to build a model that can predict the category label of a new data 
    point based on its features.
    \n**K-Nearest Neighbors (KNN):**
    KNN is a simple yet powerful algorithm for both classification and regression tasks. 
    \n**The Diabetes Dataset:**
    The Diabetes dataset contains ten baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements for 442 diabetes patients.
    \n**KNN Classification with Diabetes:**
    \n1. **Training:**
    * The KNN algorithm stores the entire Diabetes dataset (features and labels) as its training data.
    \n2. **Prediction:**
    * When presented with a new patient, KNN calculates the distance (often Euclidean distance) between this patient's features and all the patients in the training data.
    * The user defines the value of 'k' (number of nearest neighbors). KNN identifies the 'k' closest data points (patients) in the training set to the new patient.
    * KNN predicts the diabetes progression for the new patient based on the majority vote among its 'k' nearest neighbors. For example, if three out of the five nearest neighbors have a high diabetes progression, the new patient is classified as having high diabetes progression.
    **Choosing 'k':**
    The value of 'k' significantly impacts KNN performance. A small 'k' value might lead to overfitting, where the model performs well on the training data but poorly on unseen data. Conversely, a large 'k' value might not capture the local patterns in the data and lead to underfitting. The optimal 'k' value is often determined through experimentation.
    \n**Advantages of KNN:**
    * Simple to understand and implement.
    * No complex model training required.
    * Effective for datasets with well-defined clusters."""
    st.write(text)

    k = st.sidebar.slider(
        label="Select the value of k:",
        min_value= 2,
        max_value= 10,
        value=5,  # Initial value
    )

    if st.button("Begin"):
        # Load the Diabetes dataset
        diabetes = datasets.load_diabetes()
        X = diabetes.data  # Features
        y = diabetes.target  # Target labels

        # KNN for supervised classification (reference for comparison)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)

        # Predict the diabetes progression for the patients
        y_pred = knn.predict(X)

        st.write('Confusion Matrix')
        cm = confusion_matrix(y, y_pred)
        st.text(cm)

        st.subheader('Performance Metrics')
        st.text(classification_report(y, y_pred))

        # Since the dataset has multiple features, plotting them against each other may not be meaningful.
        # Thus, let's select a single feature and plot it against the target (diabetes progression).

        selected_feature_index = 2  # Choose the index of the feature you want to plot against the target
        selected_feature = diabetes.feature_names[selected_feature_index]

        # Plotting the selected feature against the target
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, selected_feature_index], y, c='blue', label='Actual Diabetes Progression')
        ax.scatter(X[:, selected_feature_index], y_pred, c='red', label='Predicted Diabetes Progression')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Diabetes Progression')
        ax.set_title(f'Visualization of Diabetes Dataset: {selected_feature} vs Diabetes Progression')
        ax.legend()
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    app()
