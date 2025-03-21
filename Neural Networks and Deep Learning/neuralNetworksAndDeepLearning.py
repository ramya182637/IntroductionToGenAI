import os
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Set seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Define dataset directory
dataset_dir = 'PetImages'
categories = ['Cat', 'Dog']

image_paths, image_labels = [], []

# Function to check if an image file is corrupted
def validate_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

# Load and filter dataset
for label in categories:
    category_path = os.path.join(dataset_dir, label)
    for img_file in os.listdir(category_path):
        img_path = os.path.join(category_path, img_file)
        if validate_image(img_path):
            image_paths.append(img_path)
            image_labels.append(label)

image_paths = np.array(image_paths)
image_labels = np.array(image_labels)

# Save cleaned dataset
np.savez('filtered_pet_data.npz', file_paths=image_paths, labels=image_labels)

# Split dataset into train, validation, and test sets
X_train, X_interim, y_train, y_interim = train_test_split(image_paths, image_labels, test_size=0.3, random_state=random_seed)
X_val, X_test, y_val, y_test = train_test_split(X_interim, y_interim, test_size=0.5, random_state=random_seed)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Function to generate image batches
def generate_batches(file_paths, labels, img_size=(150, 150), batch=32, augment=False):
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20 if augment else 0,
        width_shift_range=0.2 if augment else 0,
        height_shift_range=0.2 if augment else 0,
        shear_range=0.2 if augment else 0,
        zoom_range=0.2 if augment else 0,
        horizontal_flip=augment,
        fill_mode='nearest'
    )
    
    return data_gen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': file_paths, 'class': labels}),
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch,
        class_mode='binary',
        shuffle=True
    )

# Create generators
train_gen = generate_batches(X_train, y_train, augment=True)
val_gen = generate_batches(X_val, y_val)
test_gen = generate_batches(X_test, y_test)

# Define CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5, seed=random_seed),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

# Train the model
history = cnn_model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // train_gen.batch_size,
    validation_data=val_gen,
    validation_steps=val_gen.samples // val_gen.batch_size,
    epochs=20
)

# Save trained model
cnn_model.save('pet_classifier_v1.keras')

# Evaluate on test set
y_pred_probs = cnn_model.predict(test_gen)
y_preds = (y_pred_probs > 0.5).astype(int)
y_actual = test_gen.labels

# Compute evaluation metrics
accuracy = accuracy_score(y_actual, y_preds)
precision = precision_score(y_actual, y_preds)
recall = recall_score(y_actual, y_preds)
f1 = f1_score(y_actual, y_preds)
roc_auc = roc_auc_score(y_actual, y_pred_probs)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_actual, y_pred_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
