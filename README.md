Project Overview

This repository provides a MATLAB implementation of a system for both Gender Recognition and Speaker Recognition using a Multinomial Logistic Regression classifier. The project demonstrates a complete pattern recognition pipeline for classifying audio data based on acoustic features.

The system performs two key classification tasks:

Gender Recognition: Classifying a speaker as either male or female.

Speaker Recognition: Identifying a specific individual from a known group of speakers.

The core methodology involves:

Feature Extraction: Processing audio signals to extract key acoustic features that characterize a speaker's voice (e.g., pitch, formants, etc.).

Data Partitioning: Splitting the feature-extracted dataset into training and testing sets to ensure robust model evaluation.

Model Training: Training a Multinomial Logistic Regression model on the training data.

Evaluation: Assessing the performance of the trained classifier on the unseen test data and calculating its accuracy for both recognition tasks.
