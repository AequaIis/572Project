# Titanic Survival Prediction: Comprehensive Analysis Report

## Executive Summary

This report documents a comprehensive machine learning analysis conducted on the Titanic dataset to predict passenger survival. The project employed four different supervised machine learning approaches: **Logistic Regression**, **Support Vector Machines (SVM)**, **Random Forests and Ensemble Methods**, and **Linear/Quadratic Discriminant Analysis (LDA/QDA)**. Each approach was independently developed and optimized by team members, providing diverse perspectives on the problem. The analysis demonstrates that tree-based methods and well-tuned ensemble models consistently outperform linear approaches on this dataset.

---

## 1. Project Overview

### 1.1 Objective
Create an efficient, performant machine learning model to predict Titanic passenger survival while emphasizing model simplicity and interpretability. The project required:
- Reproducible code with random seed of 42
- 25% test set with stratified sampling
- At least 3 different supervised learning methods
- Comprehensive evaluation metrics and visualizations

### 1.2 Dataset Description
The augmented Titanic dataset contains 891 passengers with 24 features including:
- **Demographics**: Age, Sex, Passenger Class (Pclass)
- **Family Information**: SibSp (siblings/spouses), Parch (parents/children), family_size, is_alone
- **Ticket Information**: Fare, fare_per_person, ticket_group_size
- **Derived Features**: name_length, title_group, cabin_deck, age_fare_ratio, cabin_score, booking_reference

### 1.3 Data Preprocessing (Common Across All Analyses)
All team members implemented identical preprocessing steps for consistency:
1. **Missing Value Imputation**:
   - Age: Filled with median value
   - Embarked: Filled with mode value
   
2. **Feature Engineering**:
   - Dropped unnecessary columns: Cabin (too many unknowns), cabin_room_number, title, PassengerId, Name, Ticket
   - Encoded Sex as binary: male=0, female=1
   - One-hot encoded categorical variables: Embarked, Pclass, title_group, cabin_deck
   
3. **Data Splitting**:
   - Test size: 25% (stratified by Survived)
   - Random state: 42
   - Training set: 668 samples, Test set: 223 samples

4. **Feature Scaling**:
   - StandardScaler applied (essential for distance-based and linear methods)
   - Fitted on training set, transformed on test set

---

## 2. Model 1: Logistic Regression (Austin's Analysis)

### 2.1 Model Overview
**Why Logistic Regression?** It's a simple, interpretable linear model that provides probabilistic predictions and serves as a baseline for comparison with more complex methods.

### 2.2 Feature Selection
Selected 20 features balancing model complexity with predictive power:
- Numerical features: Age, SibSp, Parch, Fare, family_size, is_alone, fare_per_person, age_fare_ratio, name_length, ticket_group_size
- Encoded categorical features: Sex_binary, cabin_deck, cabin_score, booking_reference, service_id
- One-hot encoded: class_2, class_3, embarked_Q, embarked_S, plus title and cabin deck dummy variables

### 2.3 Hyperparameter Tuning
Applied GridSearchCV with cross-validation to optimize:
- **Regularization strength (C)**: [0.001, 0.01, 0.1, 1, 10, 100]
- **Penalty type**: L1 and L2 regularization
- **Solver**: liblinear, saga
- **Class weight**: None and 'balanced' (to handle class imbalance)
- **Scoring metric**: ROC-AUC (more suitable for imbalanced classification)

### 2.4 Performance Results
```
BASELINE MODEL:
- Training Accuracy: 0.8053
- Test Accuracy: 0.7937
- Precision: 0.7862
- Recall: 0.8108
- F1-Score: 0.7984
- ROC-AUC: 0.8624

AFTER HYPERPARAMETER TUNING:
- Test Accuracy: ~0.8000 (slight improvement)
- ROC-AUC improved with balanced class weights
```

### 2.5 Model Insights
1. **Top Features** (by coefficient magnitude):
   - fare_per_person: Strong positive predictor of survival
   - Sex_binary (Female): Strong positive predictor
   - class_2, class_3: Negative predictors (lower classes had lower survival)
   - Age: Negative predictor (younger passengers more likely to survive)

2. **Cross-Validation Results**:
   - 5-fold CV Accuracy: ~0.79 ± 0.03
   - 5-fold CV ROC-AUC: ~0.86 ± 0.02
   - Indicates stable model performance

3. **Confusion Matrix Analysis**:
   - True Negatives: High (correctly identified non-survivors)
   - False Negatives: Moderate (missed some survivors)
   - Well-balanced precision-recall trade-off

### 2.6 Advantages & Limitations
**Advantages:**
- Highly interpretable with clear feature coefficients
- Fast training and prediction
- Probabilistic outputs useful for decision-making
- Good ROC-AUC score (0.86)

**Limitations:**
- Assumes linear relationship between features and log-odds
- May underperform complex non-linear patterns
- Moderate accuracy (~80%)

---

## 3. Model 2: Support Vector Machines (SVM)

### 3.1 Model Overview
**Why SVM?** Effective for binary classification with strong mathematical foundation and ability to capture non-linear patterns through kernel tricks.

### 3.2 Kernel Comparison

#### 3.2.1 Linear Kernel SVM
**Hyperparameter Tuning:**
- C parameter range: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- 5-fold cross-validation

**Performance:**
```
- Best C: Identified through GridSearchCV
- Training Accuracy: 0.8020
- Test Accuracy: 0.7937
- Precision: 0.8000
- Recall: 0.7838
- F1-Score: 0.7918
- Cross-Validation Score: High consistency
```

#### 3.2.2 RBF Kernel SVM
**Hyperparameter Tuning:**
- C parameter: [0.1, 1, 10, 100, 1000]
- Gamma parameter: ['scale', 'auto', 0.001, 0.01, 0.1, 1]
- 5-fold cross-validation

**Performance:**
```
- Best C & Gamma: Optimized combination identified
- Training Accuracy: 0.8308
- Test Accuracy: 0.8138
- Precision: 0.8148
- Recall: 0.8108
- F1-Score: 0.8128
- Improved generalization over linear kernel
```

#### 3.2.3 Improved RBF SVM (Enhanced Tuning)
**Advanced Hyperparameter Tuning:**
- Expanded C range: [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
- Extended gamma range: [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
- Class weight: None and 'balanced'
- 10-fold cross-validation
- F1-score optimization (better for imbalanced data)

**Performance:**
```
- Test Accuracy: 0.8230
- Precision: 0.8301
- Recall: 0.8108
- F1-Score: 0.8203
- Balanced performance across metrics
```

### 3.3 Kernel Selection Analysis
**Best Performer: RBF Kernel with Improved Tuning**
- Linear kernel unable to capture non-linear decision boundaries
- RBF kernel superior due to its ability to map features to higher-dimensional space
- Test accuracy improvement: 0.7937 (linear) → 0.8230 (improved RBF)

### 3.4 Model Insights
1. **Decision Boundary**: RBF kernel creates non-linear boundaries, capturing complex passenger survival patterns
2. **Regularization**: Higher C values allowed flexibility; balanced with regularization to prevent overfitting
3. **Class Balance**: Including 'balanced' class weight helped address Titanic survival imbalance (61.6% survived)

### 3.5 Advantages & Limitations
**Advantages:**
- Excellent with non-linear patterns
- Effective for binary classification
- Strong test accuracy (0.823)
- Robust with proper hyperparameter tuning

**Limitations:**
- Computational complexity higher than logistic regression
- Hyperparameter tuning requires extensive grid search
- Less interpretable than linear models
- Can be sensitive to feature scaling

---

## 4. Model 3: Tree-Based Ensemble Methods (Peter's Analysis)

### 4.1 Models Evaluated
Peter's analysis compared five tree-based approaches:

#### 4.1.1 Decision Tree
```
Configuration: max_depth=10, min_samples_split=20, min_samples_leaf=10
- Training Accuracy: 0.8383
- Test Accuracy: 0.7977
- Precision: 0.8108
- Recall: 0.8108
- F1-Score: 0.8108
```
**Analysis**: Prone to overfitting (training > test accuracy). Feature selection constraints help regularization.

#### 4.1.2 Random Forest
```
Configuration: n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=10
- Training Accuracy: 0.8635
- Test Accuracy: 0.8183
- Precision: 0.8301
- Recall: 0.7927
- F1-Score: 0.8111
```
**Analysis**: Ensemble reduces overfitting. Strong test performance with stable predictions.

#### 4.1.3 Gradient Boosting
```
Configuration: n_estimators=100, learning_rate=0.1, max_depth=5
- Training Accuracy: 0.8533
- Test Accuracy: 0.8273
- Precision: 0.8387
- Recall: 0.8108
- F1-Score: 0.8245
```
**Analysis**: Sequential boosting emphasizes misclassified samples. Best F1-score among tree methods.

#### 4.1.4 Bagging Classifier
```
Configuration: n_estimators=100 (Decision Tree base estimator)
- Training Accuracy: 0.8620
- Test Accuracy: 0.8184
- Precision: 0.8387
- Recall: 0.7838
- F1-Score: 0.8108
```
**Analysis**: Parallel ensemble reduces variance. Performance similar to Random Forest.

#### 4.1.5 AdaBoost & Other Boosters
Additional ensemble methods tested to explore performance range.

### 4.2 Performance Comparison Summary
```
MODEL RANKINGS (by Test Accuracy):
1. Gradient Boosting: 0.8273 ✓ Best
2. Random Forest: 0.8183
3. Bagging: 0.8184
4. Decision Tree: 0.7977
```

### 4.3 Feature Importance Analysis (Random Forest)
Top predictive features identified:
1. **Fare**: Travel cost indicates social class and value
2. **Age**: Strong predictor of survival; women and children prioritized
3. **Sex**: Highest priority in evacuation ("women and children first")
4. **Pclass**: Travel class directly related to deck location and evacuation access
5. **Family metrics**: is_alone, family_size indicate likelihood of helping behavior

### 4.4 Overfitting Analysis
```
Train vs Test Accuracy Gap:
- Decision Tree: 4.1% gap (overfitting)
- Random Forest: 4.5% gap (well-regularized)
- Gradient Boosting: 2.6% gap (best generalization)
- Bagging: 4.4% gap
```
**Insight**: Gradient Boosting achieves best generalization; its adaptive approach better prevents overfitting.

### 4.5 Advantages & Limitations
**Advantages:**
- Non-parametric; capture complex patterns
- Feature importance readily available
- Strong test accuracy (0.827 GB)
- Handle non-linear relationships naturally
- No feature scaling required
- Robust to outliers

**Limitations:**
- More prone to overfitting (mitigated by hyperparameters)
- Less interpretable than linear models
- Hyperparameter tuning more complex
- Requires more computation than simple methods

---

## 5. Model 4: Linear & Quadratic Discriminant Analysis (Preethi's Analysis)

### 5.1 Model Overview
**Why LDA/QDA?** These are powerful statistical methods that assume multivariate normal distribution and handle multi-class problems naturally. LDA assumes equal covariance matrices; QDA relaxes this assumption.

### 5.2 Linear Discriminant Analysis (LDA)

**Model Configuration:**
- Solver: 'svd' (Singular Value Decomposition)
- Shrinkage: None
- n_components: None

**Performance:**
```
- Training Accuracy: 0.8053
- Test Accuracy: 0.8030
- Precision: 0.8000
- Recall: 0.8228
- F1-Score: 0.8113
- ROC-AUC: 0.8763
```

**Characteristics:**
1. **Decision Boundaries**: Linear; assumes equal variance across classes
2. **Discriminant Functions**: 1 (for binary classification)
3. **Model Simplicity**: Very interpretable with low computational cost

### 5.3 Quadratic Discriminant Analysis (QDA)

**Model Configuration:**
- Regularization parameter: 0.0
- Store covariance: True

**Performance:**
```
- Training Accuracy: 0.8203
- Test Accuracy: 0.8094
- Precision: 0.8333
- Recall: 0.7927
- F1-Score: 0.8126
- ROC-AUC: 0.8656
```

**Characteristics:**
1. **Decision Boundaries**: Quadratic; allows different covariances per class
2. **Model Complexity**: More parameters than LDA
3. **Flexibility**: Better fits complex separations but may overfit

### 5.4 LDA vs QDA Comparison

```
COMPARATIVE ANALYSIS:
┌─────────────┬──────────┬──────────┐
│ Metric      │   LDA    │   QDA    │
├─────────────┼──────────┼──────────┤
│ Test Acc    │  0.8030  │  0.8094  │
│ Precision   │  0.8000  │  0.8333  │
│ Recall      │  0.8228  │  0.7927  │
│ F1-Score    │  0.8113  │  0.8126  │
│ ROC-AUC     │  0.8763  │  0.8656  │
└─────────────┴──────────┴──────────┘
Winner: LDA (slight edge in AUC and recall)
```

### 5.5 Model Insights
1. **ROC-AUC Excellence**: Both models achieve ~0.87 ROC-AUC, indicating excellent ranking ability
2. **Class Separation**: Well-separated classes with respect to discriminant functions
3. **Prior Probabilities**: 
   - Class 0 (Not Survived): ~0.384
   - Class 1 (Survived): ~0.616
   
4. **Feature Covariance**: Regularization parameters balanced to prevent singular matrices

### 5.6 Advantages & Limitations
**Advantages:**
- Theoretically grounded (assumes multivariate normal distribution)
- Fast prediction due to closed-form solutions
- Excellent ROC-AUC scores (0.876 LDA)
- Natural probabilistic interpretation
- Low computational cost
- Good generalization

**Limitations:**
- Linear assumption may not hold for all feature combinations
- Assumption of multivariate normality may be violated
- QDA requires more parameters, risk of overfitting
- Moderate accuracy (~81%)

---

## 6. Comprehensive Model Comparison

### 6.1 Performance Summary Table

```
╔════════════════════════════╦════════╦════════╦═══════╦════╦═════╗
║ Model                      ║ Train  ║ Test   ║ Prec  ║Rec ║ F1  ║
╠════════════════════════════╬════════╬════════╬═══════╬════╬═════╣
║ Logistic Regression        ║ 0.8053 ║ 0.7937 ║ 0.786 ║0.81║0.798║
║ SVM Linear                 ║ 0.8020 ║ 0.7937 ║ 0.800 ║0.78║0.792║
║ SVM RBF (Basic)            ║ 0.8308 ║ 0.8138 ║ 0.815 ║0.81║0.813║
║ SVM RBF (Improved)         ║ ~0.83  ║ 0.8230 ║ 0.830 ║0.81║0.820║
║ Decision Tree              ║ 0.8383 ║ 0.7977 ║ 0.811 ║0.81║0.811║
║ Random Forest              ║ 0.8635 ║ 0.8183 ║ 0.830 ║0.79║0.811║
║ Gradient Boosting          ║ 0.8533 ║ 0.8273 ║ 0.839 ║0.81║0.824║ ⭐
║ Bagging Classifier         ║ 0.8620 ║ 0.8184 ║ 0.839 ║0.78║0.811║
║ LDA                        ║ 0.8053 ║ 0.8030 ║ 0.800 ║0.82║0.811║
║ QDA                        ║ 0.8203 ║ 0.8094 ║ 0.833 ║0.79║0.813║
╚════════════════════════════╩════════╩════════╩═══════╩════╩═════╝
```

### 6.2 Best Model Recommendation

**🏆 WINNER: Gradient Boosting**

**Why?**
1. **Highest Test Accuracy**: 0.8273 (82.73%)
2. **Best F1-Score**: 0.8245 (balanced precision-recall)
3. **Superior Generalization**: Only 2.6% train-test gap (best among all models)
4. **Strong Precision**: 0.8387 (minimizes false alarms)
5. **Balanced Recall**: 0.8108 (catches most true positives)

### 6.3 Alternative Top Performers
1. **SVM RBF (Improved)**: 0.8230 accuracy, excellent precision, good generalization
2. **Random Forest**: 0.8183 accuracy, strong feature interpretability
3. **Bagging**: 0.8184 accuracy, computationally efficient

### 6.4 By Use Case

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| **Maximum Accuracy** | Gradient Boosting | 0.8273 test acc |
| **Interpretability** | Logistic Regression | Clear coefficients |
| **Feature Importance** | Random Forest | Built-in importance scores |
| **Probability Calibration** | LDA/QDA | ROC-AUC: 0.876/0.866 |
| **Fast Prediction** | LDA | Closed-form solution |
| **Robustness** | Random Forest | Ensemble stability |

---

## 7. How Each Model Works

### 7.1 Logistic Regression
**Mathematical Foundation:**
$$P(\text{Survived}=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + ... + \beta_p X_p)}}$$

**Key Concepts:**
- Linear combination of features mapped through sigmoid function
- Produces probability between 0 and 1
- Threshold (default 0.5) determines class prediction
- Coefficients indicate feature impact on log-odds

**Decision Process:**
1. Compute weighted sum of features
2. Apply sigmoid transformation
3. If probability > 0.5, predict survived
4. Otherwise, predict did not survive

### 7.2 Support Vector Machines (SVM)
**Mathematical Foundation:**
- **Linear**: Finds optimal separating hyperplane maximizing margin
- **RBF Kernel**: Non-linearly maps features to higher-dimensional space using Gaussian function

$$K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$$

**Key Concepts:**
- Support vectors: critical boundary points
- C parameter: trades off margin width vs training accuracy
- Gamma parameter: controls influence of single training example
- Dual formulation allows kernel trick

**Decision Process:**
1. Find non-linear boundary in original feature space
2. Classify based on which side of boundary instance lies
3. Distance-weighted by support vectors

### 7.3 Decision Trees & Ensemble Methods

#### Decision Tree
**Process:**
1. Recursively partition feature space based on information gain (entropy reduction)
2. Each node represents feature threshold decision
3. Each leaf represents class prediction
4. Classification follows path from root to leaf

#### Random Forest
**Process:**
1. Build multiple independent decision trees
2. Each tree trained on random feature subset and bootstrap sample
3. Final prediction = majority vote of all trees
4. Reduces overfitting through aggregation

#### Gradient Boosting
**Process:**
1. Start with simple model (tree of depth 1-5)
2. Compute residuals (errors) from previous model
3. Fit new tree to residuals
4. Update predictions by weighted sum
5. Repeat until convergence or max iterations
6. Final prediction = sum of all sequential trees

**Why it excels:**
- Adaptively focuses on hardest-to-classify instances
- Sequential approach captures complex patterns
- Learning rate controls update magnitude

#### Bagging
**Process:**
1. Create multiple bootstrap samples from training data
2. Train independent model on each sample
3. Average predictions (regression) or majority vote (classification)
4. Reduces variance through averaging

### 7.4 Linear Discriminant Analysis (LDA)
**Mathematical Foundation:**
- Assumes features are multivariate normal within each class
- Models $P(X|Y=k) \times P(Y=k)$ using Bayes' theorem
- Discriminant function: $\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)$

**Key Concepts:**
- Common covariance matrix across classes
- Simple linear boundary separates classes
- Prior probabilities incorporated
- Class means and shared variance estimated from data

**Decision Process:**
1. Compute discriminant function for each class
2. Assign observation to class with highest discriminant value
3. Produces probability estimates via Bayes formula

### 7.5 Quadratic Discriminant Analysis (QDA)
**Mathematical Foundation:**
- Relaxes equal covariance assumption
- Each class has own covariance matrix
- Discriminant function: $\delta_k(x) = -\frac{1}{2} \log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1}(x - \mu_k) + \log(\pi_k)$

**Key Concepts:**
- Flexible quadratic boundaries
- Allows different variances per class
- More parameters → risk of overfitting
- Better for complex class separations

**Decision Process:**
1. Compute quadratic discriminant function per class
2. Assign to highest discriminant value class
3. Probabilities from quadratic posteriors

---

## 8. Key Findings & Insights

### 8.1 Feature Importance Consensus
Across all models, key predictive features:

**Tier 1 (Highest Importance):**
1. **Fare**: Travel cost indicates social class and deck location
2. **Sex/Gender**: "Women and children first" evacuation policy
3. **Pclass**: Passenger class determines evacuation priority

**Tier 2 (Important):**
4. **Age**: Children prioritized; age-fare ratio predictive
5. **Family Metrics**: is_alone, family_size, SibSp
6. **Embarked Port**: Correlated with social class

**Tier 3 (Moderate):**
7. **Name-derived features**: name_length, title_group
8. **Cabin Information**: cabin_deck, cabin_score

### 8.2 Class Balance Impact
- Dataset: 61.6% survived, 38.4% did not
- Imbalance handled by:
  - Logistic Regression: class_weight='balanced'
  - SVM: class_weight='balanced' in improved version
  - Tree methods: naturally handle imbalance
  - LDA/QDA: incorporate prior probabilities

### 8.3 Generalization Patterns

```
Model Category Performance:
Linear Models:     ~79-80% (good baseline)
Kernel Methods:    ~81-82% (good non-linearity capture)
Tree Ensembles:    ~82-83% (best overall)
Discriminant:      ~80-81% (statistical, robust)
```

**Key Observation**: Tree-based ensembles consistently outperform linear approaches, suggesting Titanic survival patterns are inherently non-linear.

### 8.4 Overfitting Analysis
```
Train-Test Gap:
SVM RBF: 1.0-1.3% (excellent generalization)
Gradient Boosting: 2.6% (best among trees)
LDA: 0.23% (near-perfect generalization)
Random Forest: 4.5% (moderate generalization)
Decision Tree: 4.1% (notable gap despite regularization)
```

### 8.5 Computational Efficiency
1. **Fastest**: LDA/QDA (closed-form solutions)
2. **Fast**: Logistic Regression, Linear SVM
3. **Moderate**: Random Forest, Decision Tree
4. **Slowest**: Gradient Boosting, RBF SVM (hyperparameter tuning)

---

## 9. Model Selection Justification

### 9.1 Why Different Models?
**Educational Value:**
- Logistic Regression: Foundation understanding of classification
- SVM: Kernel methods and margin maximization
- Tree Methods: Recursive partitioning and ensemble learning
- LDA/QDA: Statistical, probabilistic approach

**Practical Value:**
- Provides multiple solutions to same problem
- Identifies most robust approaches
- Highlights method-specific strengths and weaknesses
- Enables informed production selection

### 9.2 Production Recommendation

**Primary Model: Gradient Boosting**
- Highest accuracy and F1-score
- Best train-test generalization
- Robust to hyperparameter variations
- Handles non-linear patterns effectively

**Backup Models:**
1. **SVM RBF**: If extreme interpretability needed; excellent precision
2. **Random Forest**: If real-time predictions critical; interpretable feature importance
3. **Logistic Regression**: If explainability paramount; coefficients are clear

---

## 10. Challenges & Solutions

### 10.1 Data Challenges

| Challenge | Solution |
|-----------|----------|
| Missing Age values (19.9%) | Median imputation (preserves distribution) |
| Missing Embarked (0.2%) | Mode imputation (minimal impact) |
| Cabin column (extensive missingness) | Dropped entirely; extracted deck from non-null values |
| Class imbalance (61% survived) | Used stratified split; applied class weights in models |

### 10.2 Modeling Challenges

| Challenge | Solution |
|-----------|----------|
| Feature scaling for linear models | StandardScaler applied consistently |
| Hyperparameter tuning complexity | GridSearchCV with cross-validation |
| Model selection among many options | Comprehensive performance comparison |
| Overfitting in tree models | Depth/split/leaf regularization parameters |
| Non-linear patterns | SVM kernels and tree-based ensembles capture them |

### 10.3 Reproducibility

**Measures Taken:**
- Fixed random_state=42 across all operations
- Stratified train-test split ensures consistent distribution
- Documented hyperparameters for all models
- Seed set before data splitting, model initialization, and GridSearchCV

---

## 11. Recommendations for Future Work

### 11.1 Model Improvements
1. **Ensemble of Ensembles**: Stack Gradient Boosting predictions with other models
2. **Advanced Feature Engineering**: Create interaction terms (e.g., Sex × Age)
3. **Threshold Optimization**: Adjust classification threshold for business objectives
4. **Calibration**: Use CalibratedClassifierCV for better probability estimates

### 11.2 Data Collection
1. Historical passenger manifest validation
2. Additional contextual features (crew information, ship design)
3. Temporal patterns (time of day evacuation occurred)

### 11.3 Extended Analysis
1. **Explain predictions**: Use SHAP or LIME for instance-level explanations
2. **Fairness analysis**: Check for demographic bias in predictions
3. **Robustness testing**: Adversarial examples and perturbation analysis
4. **Class-specific analysis**: Separate models for different passenger groups

---

## 12. Conclusion

This comprehensive analysis explored five distinct machine learning approaches to predict Titanic passenger survival:

1. **Logistic Regression** (Austin): Interpretable baseline achieving 79.4% accuracy
2. **Support Vector Machines** (SVM): Non-linear boundary learning reaching 82.3% accuracy
3. **Ensemble Tree Methods** (Peter): Gradient Boosting as top performer with 82.7% accuracy
4. **Discriminant Analysis** (Preethi): Statistical approach with 80.3% accuracy and 0.876 ROC-AUC

**Key Achievements:**
- ✅ All models significantly outperform random guessing (50%)
- ✅ Best model (Gradient Boosting) achieves 82.73% test accuracy
- ✅ Models demonstrate robust generalization with minimal overfitting
- ✅ Feature importance analysis provides actionable business insights
- ✅ Reproducible code with seed=42 ensures consistency

**Primary Finding**: Tree-based ensemble methods, particularly Gradient Boosting, are optimal for this Titanic classification task due to their ability to capture non-linear survival patterns while maintaining excellent generalization.

The diversity of approaches taken demonstrates that machine learning problems rarely have one "correct" solution—instead, comparing multiple methodologies provides confidence in findings and identifies the most appropriate technique for production deployment.

---

## Appendices

### Appendix A: Dataset Statistics

```
Dataset Shape: 891 passengers × 24 features
Survival Distribution:
  - Survived: 549 (61.6%)
  - Did not survive: 342 (38.4%)

Train Set: 668 samples (25% = 223 test samples)
Feature Types:
  - Numerical: 13
  - Categorical: 11
  
Missing Values:
  - Age: 177 (19.9%)
  - Embarked: 2 (0.2%)
  - Others: None or handled
```

### Appendix B: Hyperparameter Grid Ranges

**Logistic Regression:**
- C: [0.001, 0.01, 0.1, 1, 10, 100]
- penalty: [l1, l2]
- solver: [liblinear, saga]
- class_weight: [None, balanced]

**SVM RBF:**
- C: [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
- gamma: [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, scale, auto]
- class_weight: [None, balanced]

**Gradient Boosting:**
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 5
- min_samples_split: 20
- min_samples_leaf: 10

### Appendix C: Feature List (Final Dataset)

```python
Feature Count: 41 features
- Numerical: Age, SibSp, Parch, Fare, name_length, 
             family_size, is_alone, ticket_group_size,
             fare_per_person, age_fare_ratio, cabin_score,
             booking_reference, service_id
- Categorical (encoded): Sex, Embarked, Pclass, title_group,
                        cabin_deck
```

---

**Report Compiled**: February 1, 2026  
**Project**: DATA 572 - Titanic Survival Prediction  
**Team Members**: Austin, Peter, Preethi  
**Analysis Duration**: Comprehensive multi-method comparison
