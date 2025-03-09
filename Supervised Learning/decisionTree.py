import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Part 1: Building the Decision Tree

# Task 1.1: Choosing a dataset
df = pd.read_csv('weatherHistory_data.csv')[:600]
df = df.drop(columns=['Precip Type', 'Formatted Date', 'Apparent Temperature (C)', 
                      'Wind Bearing (degrees)', 'Visibility (km)', 'Daily Summary', 'Loud Cover'])
print("Columns in dataset:", df.columns)

# Part 1.2: Preprocessing of the data

# Handling missing values by filling them with the mean for numerical columns
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encoding categorical variables using LabelEncoder 
label_encoder = LabelEncoder()
df['Summary'] = label_encoder.fit_transform(df['Summary'])

# Normalizing numerical features 
scaler = StandardScaler()
df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']] = scaler.fit_transform(df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']])

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['Summary'])
y = df['Summary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Part 1.3: Training the decision tree

# Task 2.2: Adjusting model parameters to address class imbalance and prevent overfitting/underfitting
model = DecisionTreeClassifier(max_depth=20, min_samples_split=30, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Part 2: Debugging Issues

# Task 2.1: Testing for overfitting or underfitting

# Making predictions on the training and test set
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculating accuracy for training and test set
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Task 2.2: Debug common issues (i.e., precision and recall)

# Calculating Precision and Recall
precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_test_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Confusion Matrix: To understand prediction errors
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)

# Ploting confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Part 3: Cross-validation to evaluate the model performance

# Performing cross-validation to get more robust performance metrics
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Average cross-validation score: {scores.mean()}")

# Optional: Trying Random Forest or other models
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Test Accuracy: {rf_accuracy}")
