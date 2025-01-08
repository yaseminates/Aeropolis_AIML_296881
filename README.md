# Aeropolis_AIML_296881

---
## [Section 0] Title and Team Members
- Project Title: Aeropolis
- Group Number: 2
- Team Members:
  - 296881 Guia Ludovica Basso (Captain)
  - 297061 Alessio Giannotti
  - 304011 Yasemin Ateş
    
This is a group project for the Artificial Intelligence and Machine Learning course of LUISS University (a.y. 2024-2025).

---
## [Section 1] Introduction

**Project Overview:** In the futuristic city of Aeropolis, autonomous delivery drones are crucial for the efficient transportation of goods. This project aims to predict the cargo capacity of these drones per flight, considering various environmental and operational factors that affect drone performance. By understanding the influence of conditions such as weather, terrain, and equipment, we aim to improve the operational efficiency of these drones.

---
## [Section 2] Methods
The analysis begins with exploratory data analysis (EDA) to clean the data, handle missing values, identify outliers, and analyze the target variable (cargo capacity). Then we move on to apply and compare machine learning (ML) algorithms to predict cargo capacity.

### Proposed Ideas:
We developed a machine learning model to predict the cargo capacity of delivery drones based on a dataset containing various features. The features include weather conditions, terrain type, and equipment specifications. We applied several models to identify the best one for predicting cargo capacity:

- Random Forest
- Linear Regression
- Gradient Boosting
- XGBoost

We utilized a sample dataset to speed up the experimentation process and reduce computation time, while fully utilizing the computational resources at hand.

### **Environment:** The model was developed using the Python programming language and the following libraries:

- **pandas** for data manipulation
- **numpy** for numerical operations
- **scikit-learn** for machine learning algorithms
- **matplotlib** and seaborn for data visualization
- **rich** for making the output more readable and styled in the console (including tables and colorful text)
- **pyampute** for handling missing data in a structured way, including random imputation
- **math** for statistical functions and advanced calculations
- **missingno** for visualizing missing data and identifying patterns
- **scipy** for statistical functions and advanced calculations
- **warnings** to manage and filter warning messages during the process
- **statsmodels** for statistical modeling and hypothesis testing
- **joblib** for saving and loading machine learning models
- **xgboost** for implementing the XGBoost model, a gradient boosting algorithm
- **builtins** for basic Python functions like input/output handling
- **collections** for “defaultdict” (auto-creating lists for keys)
- **itertools** generates all pairs of categorical and numerical features for plotting
- **statsmodels** for power analysis to calculate the required sample size.

---
#### To replicate our environment, follow these instructions:
1. Make sure you have cloned the repository by having ran:
    ```
    git clone https://github.com/yaseminates/Aeropolis_AIML_296881
    ```
3. Change the directory to the cloned repository.
   ```
   cd Aeropolis_AIML_296881
   ```
5. Then run:
    ```
    conda env create -f environment.yml
    conda activate aeropolis-env
    ```
4. Your Terminal should look like this:
    ```
    (env-aeropolis) yourusername@your-machine Aeropolis_AIML_296881 %
    ```
---
## **Flowchart:**
![Flowchart in structure: Data Loading -> EDA -> Data Processing -> Data Splitting -> Model election -> Training and Validation -> Evaluation -> Final Model](images/flowchart.jpeg "Flowchart")

## **Contents of the Jupyter Notebook**
![Contents](images/contents.jpeg "Contents")











