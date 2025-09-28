# ============================================================
# 2. Install Required Libraries
# ============================================================
!pip install opencv-python-headless Pillow tqdm
!pip install mediapipe opencv-python

import numpy as np

"""**preprocessing**"""

pickle_path="/content/drive/MyDrive/DEPI-Project6/DEPI-Project6/data_augmented.pickle"
import pickle

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Check what keys are inside
print(data.keys())

"""**shapes of the data**"""

X = np.array(data['data'])
y = np.array(data['labels'])

print("Images shape:", X.shape)
print("Labels shape:", y.shape)

"""**Convert letter labels to integerss**"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Original labels (letters)
print(y[:20])  # check first 10 labels
# Fixed (use the same encoder for both training and prediction):
le = LabelEncoder()
y_int = le.fit_transform(y)  # This should be saved and reused

# Save the label encoder for webcam use
import joblib
joblib.dump(le, "/content/drive/MyDrive/SignLanguageProject/label_encoder.pkl")


print("First 10 integer labels:", y_int[:10])

"""**one hot encoding**"""

num_classes = len(np.unique(y_int))
y_encoded = to_categorical(y_int, num_classes=num_classes)

print("One-hot encoded labels shape:", y_encoded.shape)

"""**Split into train/test sets**"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_int
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

"""**model tranning**"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Build the model
model = Sequential([
    Dense(128, input_shape=(42,), activation='relu'),  # first hidden layer
    Dropout(0.3),                                     # prevent overfitting
    Dense(64, activation='relu'),                     # second hidden layer
    Dropout(0.3),
    Dense(num_classes, activation='softmax')          # output layer
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Check model summary
model.summary()

"""**train the model**"""

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64
)

"""**test accuracy**"""

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

"""**save the model**"""

# Save model
model.save("/content/drive/MyDrive/SignLanguageProject/En_model.h5")
print("Model saved successfully!")

"""**save model with pickle **"""

import pickle

with open("/content/drive/MyDrive/SignLanguageProject/En_model.p", "wb") as f:
    pickle.dump(model, f)

print("Model saved with pickle!")

"""**load the model**"""

from tensorflow.keras.models import load_model

model = load_model("/content/drive/MyDrive/SignLanguageProject/En_model.h5")
print("Model loaded successfully!")

"""**predict on sample**"""

import numpy as np

# Example: take first test sample
sample = X_test[0:1]  # shape (1, 42)
prediction = model.predict(sample)

predicted_class = np.argmax(prediction)  # integer class
predicted_letter = le.inverse_transform([predicted_class])[0]  # convert back to letter

print("Predicted letter:", predicted_letter)
print("True letter:", le.inverse_transform([np.argmax(y_test[0])])[0])

"""**predict**"""

# Take first sample from the test set
new_sample = X_test[0:1]  # already shaped (1,42)
prediction = model.predict(new_sample)

predicted_class = np.argmax(prediction)
predicted_letter = le.inverse_transform([predicted_class])[0]

print("Predicted letter:", predicted_letter)
true_letter = le.inverse_transform([np.argmax(y_test[0])])[0]
print("True letter:", true_letter)

# new_sample: your 42-feature vector
new_sample = np.array(new_sample).reshape(1, -1)
new_sample = new_sample / X.max()  # normalize
prediction = model.predict(new_sample)
predicted_class = np.argmax(prediction)
predicted_letter = le.inverse_transform([predicted_class])[0]

print("Predicted letter:", predicted_letter)

"""**visualize the accuracy **"""

import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""*some data sample*"""

for i in range(10):
    sample = X_test[i:i+1]
    pred_class = np.argmax(model.predict(sample))
    pred_letter = le.inverse_transform([pred_class])[0]
    true_letter = le.inverse_transform([np.argmax(y_test[i])])[0]
    print(f"Predicted: {pred_letter}, True: {true_letter}")
