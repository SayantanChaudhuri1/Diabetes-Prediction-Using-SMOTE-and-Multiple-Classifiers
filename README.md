# Diabetes-Prediction-Using-SMOTE-and-Multiple-Classifiers
----------------------------------------------------------
#### This study presents a machine learning-based approach for diabetes prediction using structured patient data. The dataset comprises 100,000 patient records, including key risk factors such as age, BMI, HbA1c levels, and blood glucose levels. Due to class imbalance in diabetic cases, SMOTE is applied to enhance model fairness.  Seven classifiers—Random Forest, Support Vector Machine (SVM), Logistic Regression, K-Nearest Neighbors (KNN), Naïve Bayes, Gradient Boosting, and Decision Tree—are implemented and assessed using metrics such as accuracy, precision, recall, F1-score, ROC AUC, and confusion matrices. The results indicate that ensemble models, particularly Random Forest and Gradient Boosting, outperform other classifiers in predictive performance. The findings of this study contribute to the ongoing research on machine learning applications in healthcare, demonstrating the importance of robust preprocessing techniques and classifier selection in medical diagnosis.

### INTRODUCTION 
### Diabetes mellitus is a prevalent and life-threatening metabolic disorder characterized by high blood glucose levels. Uncontrolled diabetes can lead to severe complications such as cardiovascular disease, kidney failure, nerve damage, and blindness. The growing incidence of diabetes necessitates efficient and accurate predictive models to aid early diagnosis and intervention. Traditional diagnostic methods rely on clinical evaluations and biochemical tests, which can be time-consuming and costly. Machine learning offers an innovative alternative, providing automated predictions with high accuracy and efficiency.
### However, challenges such as data imbalance can significantly affect the performance of ML models. The presence of imbalanced datasets, where diabetic cases are underrepresented compared to non-diabetic cases, can lead to biased predictions. This study tackles this issue by implementing SMOTE, a widely used oversampling technique that generates synthetic samples to balance the dataset.
### The primary objectives of this research are:

To implement and evaluate multiple ML classifiers for diabetes prediction.
To assess the impact of SMOTE in handling class imbalance.
To compare model performances based on key evaluation metrics.
To determine the most effective classifier for predicting diabetes.

#### DATASET
## Dataset - `diabetes_prediction_dataset.csv`
### The dataset used in this study comprises various demographic, medical history, and lifestyle-related attributes, including:

> Age	 -  Patient’s age in years.
> Gender - Male, female, or other
> BMI (Body Mass Index) - A measure of body fat based on height and weight.
> Hypertension - Presence (1) or absence (0) of high blood pressure.
> Heart Disease	- Presence (1) or absence (0) of cardiovascular conditions.
> Smoking History	- Past or present smoking habits.
> HbA1c Level	- Measure of blood sugar levels over the past 2-3 months.
> Blood Glucose Level- 	Instantaneous blood sugar measurement.
> Diabetes - Target variable (1 for diabetic, 0 for non-diabetic).

#### MODEL WORKFLOW
### To provide a structured overview of the approach, the following steps outline the workflow of the predictive model:
### Data Collection: Acquisition and cleaning of diabetes-related medical data.
### Data Preprocessing: Handling missing values, encoding categorical variables, and normalizing numerical features.
### Class Balancing: Applying Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance and enhance model fairness.
### Model Selection: Implementing and evaluating multiple machine learning classifiers.Seven ML classifiers were selected for evaluation: 
### Random Forest
### Support Vector Machine (SVM)
### Logistic Regression
### K-Nearest Neighbors (KNN)
### Naïve Bayes
### Gradient Boosting
### Decision Tree

### Training and Testing: Splitting the dataset into training and testing sets and fitting the models accordingly.
### Evaluation: Assessing model performance using accuracy, precision, recall, F1-score, and ROC AUC.
### Comparison and Insights: Analyzing the results to determine the best-performing model.

### RESULTS AND DISCUSSION
#### The models are trained and evaluated using various performance metrics. The detailed results are presented in the following table:

![image](https://github.com/user-attachments/assets/eda692d6-9fe3-43c8-b075-976afac1e5dd)

![image](https://github.com/user-attachments/assets/685f180b-7f82-4195-813c-aea14cb5e2f5)
