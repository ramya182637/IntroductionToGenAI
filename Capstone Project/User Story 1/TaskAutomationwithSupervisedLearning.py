import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Loading the dataset
data_frame = pd.read_csv('C:/Users/AVuser/Desktop/Genspark Assignments/Introduction to GenAI/Capstone Project/User Story 1/students_data.csv')

# Step 2: Creating a new target variable 'outcome' for classification (Pass if G3 >= 8, else Fail)
data_frame['outcome'] = ['Pass' if score >= 8 else 'Fail' for score in data_frame['G3']]

# Step 3: Defining the feature set features and target target_value
features = data_frame[['studytime', 'failures']]  #
target_value = data_frame['outcome']  

# Step 4: Splitting the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target_value, test_size=0.2, random_state=42)

# Step 5: Creating and training the RandomForestClassifier model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(features_train, target_train)

# Step 6: Making predictions on the test data
predictions = random_forest_model.predict(features_test)

# Step 7: Evaluating the model's performance
model_accuracy = accuracy_score(target_test, predictions)
print(f"Accuracy: {model_accuracy:.2f}")

# Step 8: Printing the classification report
print("Classification Report:")
print(classification_report(target_test, predictions))
