# Gradient Boosting Classifier - From Scratch

## Overview

The **Gradient Boosting Classifier** is a custom-built binary classification model that leverages decision trees as base learners. Implemented entirely from scratch without external ML libraries, this model demonstrates how boosting sequentially improves predictions using gradient descent on the log-loss function. 

Key highlights include:
- Decision Tree learner
- Gradient descent optimization
- Subsampling and feature bagging
- Gradient clipping and early stopping
- Platt scaling for probability calibration

---

## Functionality & Usage

### What does the model do?

The **Gradient Boosting Classifier** performs binary classification by building an ensemble of weak learners (regression trees), where each subsequent model is trained to correct the residual errors made by the previous ones. 

It optimizes the log-loss function using gradient descent, which makes it especially effective for:
- **High-accuracy predictive modeling**
- **Non-linear relationships and feature interactions**
- **Binary classification problems** like spam detection, customer churn, or medical diagnosis

### When should it be used?

- When you need **high-performance classification**
- For datasets with **complex patterns or interactions**
- When interpretability through **feature importance** and **tree paths** is valuable
- Suitable for **tabular data** with moderate to large size

---

## Model Testing

### How did you test your model to determine if it is working correctly?

The model was tested rigorously using the following strategies:

1. **Multiple Benchmark Datasets**  
   - `linear_separable.csv`  
   - `non_linear_moons.csv`  
   - `breast_cancer.csv`  

2. **Performance Metrics Evaluated**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
   - ROC AUC  
   - Log Loss  

3. **Automated Unit Testing with PyTest**  
   The test suite checks if the model meets expected metric thresholds and returns valid probability distributions.

4. **Validation Split for Early Stopping**  
   A holdout validation set is used to halt training when performance stops improving.

5. **Probability Calibration Check**  
   `predict_proba()` results are verified to be in the valid probability range and sum to 1.

---

## User-Tunable Parameters

### What parameters have been exposed to users for tuning performance?

| Parameter           | Description                                                  | Default |
|---------------------|--------------------------------------------------------------|---------|
| `n_estimators`      | Number of boosting rounds                                    | 200     |
| `learning_rate`     | Shrinks the contribution of each tree                        | 0.05    |
| `max_depth`         | Maximum depth of each decision tree                          | 5       |
| `min_samples_split` | Minimum samples required to split an internal node           | 2       |
| `subsample`         | Fraction of samples used per tree (for stochastic boosting)  | 0.8     |
| `max_features`      | Number of features to consider per split                     | None    |

These parameters help control model complexity, overfitting, and runtime performance.

---

## Limitations & Challenges

### Are there specific inputs that the implementation has trouble with?

Yes, the following scenarios may pose challenges:

1. **Severe Class Imbalance**  
   The model doesn't incorporate class weighting. Minor classes may be underrepresented.

2. **Very Noisy Data**  
   Even with subsampling and clipping, noise may lead to overfitting in deeper trees.

3. **Multi-Class Tasks**  
   The current version supports binary classification only.

### Given more time, could these issues be addressed?

- **Imbalanced Datasets:** Integrate class weighting or focal loss.
- **Noisy Data:** Add regularization and pruning strategies.
- **Multi-class Support:** Implement one-vs-all strategy or softmax loss extension.

---


---
## How to run the code
1. **Create a virtual environment:**
   ```cmd
   python -m venv gradient
   ```

2. **Activate the virtual environment:**
   ```cmd
   gradient\Scripts\activate
   ```

3. **Install the dependencies from the requirements.txt file:**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Run the tests from GradientBoosting/tests using:**
   ```cmd
   pytest
   ```

---
---
## How to Use the Model
from GradientBoosting import GradientBoostingClassifier

# Load your data (X: features, y: labels)
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)
model.y_train_ = y  # Required for Platt scaling
results = model.fit(X, y)

# Predict probabilities and class labels
y_proba = results.predict_proba(X)
y_pred = results.predict(X)

# Evaluate performance
metrics = results.evaluate(X, y)
print(metrics)



## Conclusion
The Gradient Boosting Classifier is a robust, interpretable model that performs well on binary classification tasks. Designed from scratch using first principles, it balances practical performance with algorithmic transparency. With enhancements such as multi-class support, better regularization, and interpretability tools, it can become a production-ready tool.


---
## Team Members
1. Jaitra Narasimha Valluri (A20553229)
2. ⁠Chandrika Rajani (A20553311)
3. ⁠Pooja Sree Kuchi (A20553325)
---




