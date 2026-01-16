# IMPORT NUMPY LIBRARY
# numpy is used for numerical operations like arrays and math
# np is a short for numpy
import numpy as np

# IMPORT PANDAS LIBRARY
# pandas is used to handle tabular data (rows & columns)
# DataFrame is similar to an Excel sheet
# pd is a short alias for pandas
import pandas as pd

# IMPORT BREAST CANCER DATASET
# load_breast_cancer() loads a ready-made labeled medical dataset
# sklearn.datasets contains built-in datasets
from sklearn.datasets import load_breast_cancer

# IMPORT TRAIN-TEST SPLIT FUNCTION
# This function splits data into training and testing sets
# sklearn.model_selection contains data splitting tools
from sklearn.model_selection import train_test_split

# IMPORT STANDARD SCALER
# StandardScaler is used to normalize feature values
# Normalization is important for distance-based algorithms like SVM
from sklearn.preprocessing import StandardScaler

# IMPORT SUPPORT VECTOR CLASSIFIER
# SVC is the SVM model used for classification problems
from sklearn.svm import SVC

# IMPORT EVALUATION METRICS
# accuracy_score → overall correctness
# classification_report → precision, recall, f1-score
# confusion_matrix → actual vs predicted comparison
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# IMPORT MATPLOTLIB
# matplotlib.pyplot is used for plotting graphs
# plt is a short alias
import matplotlib.pyplot as plt

# LOAD THE DATASET
# This returns a Bunch object (dictionary-like structure)
data = load_breast_cancer()

# EXTRACT FEATURES FROM DATASET
# data.data contains all input feature values
# Each row = one tumor sample
# Each column = one medical measurement
X = data.data

# EXTRACT TARGET LABELS
# data.target contains output labels
# 0 = Malignant, 1 = Benign
y = data.target

# CONVERT FEATURES INTO A DATAFRAME
# pd.DataFrame converts numerical data into table format
# columns=data.feature_names assigns column names
df = pd.DataFrame(X, columns=data.feature_names)

# ADD TARGET COLUMN TO DATAFRAME
# This helps us view features and labels together
df['Target'] = y

# PRINT DATASET SHAPE
# shape shows number of rows and columns
print("Dataset Shape:", df.shape)

# PRINT HEADING FOR DATA PREVIEW
# \n adds a new line
print("\nFirst 5 Rows of Dataset:\n")

# DISPLAY FIRST 5 ROWS OF DATAFRAME
# head() returns top 5 rows by default
print(df.head())

# SPLIT DATA INTO TRAINING AND TESTING SETS
# X → features
# y → labels
# test_size=0.2 means 20% test data
# random_state=42 ensures same split every run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# CREATE STANDARD SCALER OBJECT
# This object will compute mean and standard deviation
scaler = StandardScaler()

# SCALE TRAINING DATA
# fit_transform() learns scaling values AND applies them
X_train = scaler.fit_transform(X_train)

# SCALE TEST DATA
# transform() applies the SAME scaling learned from training data
# This avoids data leakage
X_test = scaler.transform(X_test)

# CREATE SVM MODEL
# kernel='linear' means a straight line (hyperplane) is used
svm_model = SVC(kernel='linear')

# TRAIN THE SVM MODEL
# fit() allows the model to learn patterns from training data
svm_model.fit(X_train, y_train)

# MAKE PREDICTIONS ON TEST DATA
# predict() outputs predicted class labels (0 or 1)
y_pred = svm_model.predict(X_test)

# PRINT MODEL ACCURACY
# accuracy_score compares predictions with actual labels
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# PRINT CLASSIFICATION REPORT
# Shows precision, recall, f1-score, and support
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# CREATE CONFUSION MATRIX
# Rows = Actual values
# Columns = Predicted values
cm = confusion_matrix(y_test, y_pred)

# CREATE FIGURE FOR CONFUSION MATRIX
# figsize controls plot size
plt.figure(figsize=(5, 4))

# DISPLAY CONFUSION MATRIX AS IMAGE
# cmap='Blues' adds blue color gradient
plt.imshow(cm, cmap='Blues')

# ADD TITLE TO PLOT
plt.title("Confusion Matrix")

# LABEL X-AXIS
plt.xlabel("Predicted")

# LABEL Y-AXIS
plt.ylabel("Actual")

# ADD COLOR BAR FOR SCALE
plt.colorbar()

# WRITE VALUES INSIDE CONFUSION MATRIX CELLS
# Outer loop → rows
# Inner loop → columns
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center', va='center')

# DISPLAY THE FINAL PLOT
plt.show()
