SREE RAM CHARAN TEJA             700762701
# Decision Tree Classifier on Iris Dataset

## 1. Task
We trained a **Decision Tree Classifier** using `scikit-learn` on the Iris dataset. The goal was to observe how the **maximum depth** of the tree affects training accuracy, test accuracy, and the risk of **underfitting vs overfitting**.  

---

## 2. Methodology
- The Iris dataset has **150 samples** of flowers with **4 features** (sepal length, sepal width, petal length, petal width).  
- We split the dataset into **70% training data** and **30% testing data**.  
- Decision Trees were trained with different values of `max_depth` = **1, 2, 3**.  
- After training, we computed the **accuracy** on both training and test sets.  

---

## 3. Python Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train with different depths
depths = [1, 2, 3]
results = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    results.append((d, train_acc, test_acc))

# Print results
print("Depth | Training Accuracy | Test Accuracy")
for d, train_acc, test_acc in results:
    print(f"{d:<5} | {train_acc:.4f}           | {test_acc:.4f}")
