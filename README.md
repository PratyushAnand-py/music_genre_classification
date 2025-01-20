# music_genre_classification
Exploring Music Genre classification using Machine Learning and leveraging Audio Processing using Deep Learning

This repository contains a Jupyter Notebook for building and evaluating a Music Classifier. The classifier is designed to categorize music tracks into predefined genres or classes using machine learning techniques.
Features

    Preprocessing Pipeline: Includes data cleaning, feature extraction, and normalization.
    Feature Engineering: Utilizes audio features such as tempo, rhythm, MFCCs, and spectral characteristics.
    Model Training: Implements machine learning algorithms such as decision trees, SVMs, or neural networks for classification.
    Evaluation Metrics: Calculates metrics like accuracy, precision, recall, and F1-score to assess model performance.
    Visualization: Generates plots for feature importance, confusion matrices, and performance metrics.

Requirements

To run the notebook, you will need the following dependencies:

    Python 3.7 or later
    Jupyter Notebook
    numpy
    pandas
    scikit-learn
    librosa
    matplotlib
    seaborn

Install the required packages using pip:

pip install numpy pandas scikit-learn librosa matplotlib seaborn

Usage

    Clone the repository and navigate to the project directory:

git clone <repository-url>
cd Music_Classifier

Open the Jupyter Notebook:

    jupyter notebook Music_classifier.ipynb

    Follow the steps in the notebook to load your dataset, preprocess the data, train the model, and evaluate its performance.

Dataset

The notebook is designed to work with audio datasets containing labeled music samples. You can use publicly available datasets like the GTZAN Music Genre Dataset or your custom dataset.

Ensure your dataset includes:

    Audio files in a supported format (e.g., WAV, MP3)
    Labels corresponding to each audio file's genre or class

Modify the data loading and preprocessing sections to fit your dataset's structure.
Output

The notebook produces the following outputs:

    Trained model for music classification
    Visualizations of model performance and audio features
    Summary of evaluation metrics

Customization

You can customize the classifier by:

    Modifying feature extraction parameters (e.g., number of MFCCs)
    Experimenting with different machine learning algorithms
    Tuning hyperparameters for optimal performance
