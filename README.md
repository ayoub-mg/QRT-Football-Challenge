# QRT Football Challenge

This project was developed as part of the **QRT Data Challenge x Télécom Business Finance**, where we tackled the task of predicting the outcomes of football matches. Our team's submission achieved **1st place on the public leaderboard** and **2nd place on the private leaderboard**, with an accuracy of 0.4861, just shy of the top private score of 0.4875.

The whole dataset and further description of the competition can be found on the [challenge website](https://challengedata.ens.fr/participants/challenges/143/).

## Project Structure
- `benchmark_notebook.ipynb`: This notebook serves as the baseline for the project. The model predicts whether the away team wins, using an XGBoost classifier. Basic data preprocessing and analysis were done, and a benchmark accuracy was established based on simple assumptions like always predicting home wins.
  
- `Final_submission.ipynb`: This is the notebook for our final submission, where multiple machine learning models were applied. We utilized **SVM**, **XGBoost**, **CATBoost**, and **Multi-Layer Perceptron (MLP)** models. We then ensembled these models to enhance the overall performance. We also carefully handled data preprocessing, dropping rows with NaN values and applying a train-validation split to ensure the robustness of the final predictions.

## Approach

### Data Preprocessing
We started by cleaning the dataset, where we dropped rows with missing values to ensure consistency. The cleaned data was then split into training and validation sets to monitor overfitting and tune hyperparameters effectively.

### Modeling
We experimented with several models during the challenge, including:

1. **SVM Model:** A Support Vector Machine was implemented to identify patterns in the data.
2. **XGBoost Model:** XGBoost was one of our key models due to its ability to handle tabular data with high accuracy.
3. **CATBoost Model:** This model was particularly useful for handling categorical data and helped improve performance.
4. **Multi-Layer Perceptron (MLP):** We integrated a neural network approach through MLP, which introduced a deep learning element to our solution.

### Ensembling
The predictions from the different models were combined using an ensemble method. This strategy allowed us to leverage the strengths of each model and significantly improve the final accuracy.

## Results

Our final model, through ensembling, was able to secure **first place on the public leaderboard** and **second place on the private leaderboard** with an accuracy of **0.4861**, which was very close to the top private accuracy of **0.4875**.
