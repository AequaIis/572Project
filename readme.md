# Titanic Survival Prediction Project

## Project Goal

The objective of this project is to create a *small yet performant model* that predicts the survival of passengers aboard the Titanic. Emphasis is placed on building an *efficient model* that balances predictive performance with model simplicity (i.e., using fewer features and simpler models where possible).


## Dataset

This dataset is an *augmented version* of the well-known Titanic dataset originally from [Data Science Dojo](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv). It contains information about passengers aboard the Titanic, including demographics, family relationships, ticket and cabin details, and additional attributes designed to support modeling.

**Columns:**

| Column Name           | Description |
|-----------------------|-------------|
| PassengerId           | Unique identifier for each passenger. |
| Survived              | Survival status (0 = No, 1 = Yes). This is the target variable. |
| Pclass                | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd). |
| Name                  | Full name of the passenger. |
| Sex                   | Gender of the passenger. |
| Age                   | Age in years. |
| SibSp                 | Number of siblings or spouses aboard. |
| Parch                 | Number of parents or children aboard. |
| Ticket                | Ticket number. |
| Fare                  | Fare paid for the ticket. |
| Cabin                 | Cabin number. |
| Embarked              | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). |
| name_length           | Number of characters in the passenger's name. |
| title_group           | Title extracted from the passenger's name (e.g., Mr, Mrs, Miss, Other). |
| family_size           | Total number of family members aboard (SibSp + Parch + 1). |
| is_alone              | Indicates if the passenger was traveling alone (1 = yes, 0 = no). |
| ticket_group_size     | Number of passengers sharing the same ticket. |
| fare_per_person       | Fare divided by the number of passengers on the same ticket. |
| age_fare_ratio        | Ratio of age to fare. |
| cabin_deck            | First character of the cabin (deck letter, e.g., A, B, C, ...). |
| cabin_room_number     | Numeric part of the cabin. |
| booking_reference     | Numeric hash generated from the ticket number. |
| service_id            | Unique numeric identifier assigned to each passenger. |
| cabin_score           | Numeric score associated with the cabin (custom-generated for this project). |
| name_word_count       | Number of words in the passenger's name. |

---

## Project Requirements

1. **Deterministic and reproducible code**  
   - Set the random seed to `42` for all operations involving randomness.

2. **Data splitting**  
   - Split the dataset into training and test sets using 25% of the data as the test set.  
   - Ensure the split is stratified by the `Survived` column.  

3. **Model selection**  
   - Choose *at least three* supervised machine learning methods from the following:
     - Logistic Regression  
     - K-Nearest Neighbors (KNN)  
     - Linear Discriminant Analysis (LDA)  
     - Decision Trees  
     - Random Forests  

4. **Model training and evaluation**  
   - Train models *only on the training set*.  
   - Evaluate models on the test set and report performance metrics.  
   - Report the *number of features* used in the final model.  

## Recommended Project Procedure

To accomplish this project, you should perform the following steps:

1. **Data splitting**
    - Split the dataset into training and test sets using a fixed seed of `42` to ensure reproducibility.
    - Keep the test set completely separate until final evaluation.

2. **Data preprocessing**  
   - For example, handle missing values, normalize or standardize numerical features, and encode categorical variables when needed.

3. **Data resampling**  
   - Use resampling techniques (e.g., k-fold cross-validation) on the training set to ensure robust evaluation and hyperparameter tuning.  

4. **Model building and training**  
   - Build and train the models using the selected methods.  
   - Consider feature importance or contribution to reduce the number of features while maintaining performance.  

5. **Hyperparameter tuning**  
    - Tune model hyperparameters on the training set (using resampling) to improve performance and generalization.

6. **Result evaluation and visualization**  
   - Evaluate each model’s performance using relevant metrics.  
   - Visualize performance results in an intuitive way (e.g., accuracy curves, comparison charts).  

7. **Analysis and discussion**  
   - Analyze and compare the performance of the chosen models.  
   - Discuss strategies used to improve performance and efficiency.  
   - Describe any challenges encountered and how they were addressed.  
