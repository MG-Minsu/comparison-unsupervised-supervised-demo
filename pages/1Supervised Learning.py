# Import libraries
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets
import matplotlib.pyplot as plt

# Define the Streamlit app
def app():
    st.subheader('Supervised Learning, Classification, and KNN with Digits Dataset')

    # Load the digits dataset
    digits = datasets.load_digits()
    X = digits.data  # Features
    y = digits.target  # Target labels (species)

    # Sidebar slider for selecting the value of k
    k = st.sidebar.slider(
        label="Select the value of k:",
        min_value=2,
        max_value=10,
        value=5,  # Initial value
    )

    if st.button("Begin"):
        # KNN for supervised classification

        # Define the KNN classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)

        # Train the KNN model
        knn.fit(X, y)

        # Predict the labels for the data
        y_pred = knn.predict(X)

        # Display confusion matrix
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y, y_pred)
        st.text(cm)

        # Display performance metrics
        st.subheader('Performance Metrics')
        st.text(classification_report(y, y_pred))

        # Plotting
        unique_labels = np.unique(y_pred)
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        fig, ax = plt.subplots(figsize=(8, 6))

        for label, color in zip(unique_labels, colors):
            indices = y_pred == label
            ax.scatter(X[indices, 0], X[indices, 1], label=label, c=color)

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('Visualization of Handwritten Digits Dataset')

        ax.legend()
        ax.grid(True)

        # Display the plot within Streamlit app
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    app()
