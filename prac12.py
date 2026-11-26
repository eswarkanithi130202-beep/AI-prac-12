from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load toy dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Create models
knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)
dt = DecisionTreeClassifier(random_state=42)

# Train models
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Make predictions
knn_pred = knn.predict(X_test)
dt_pred = dt.predict(X_test)

# Compare accuracy
knn_acc = accuracy_score(y_test, knn_pred)
dt_acc = accuracy_score(y_test, dt_pred)

print("Model Comparison")
print("----------------")
print("KNN Accuracy:", round(knn_acc * 100, 2), "%")
print("Decision Tree Accuracy:", round(dt_acc * 100, 2), "%")

if knn_acc > dt_acc:
    print("Best Model: KNN")
elif dt_acc > knn_acc:
    print("Best Model: Decision Tree")
else:
    print("Both models have equal accuracy")
