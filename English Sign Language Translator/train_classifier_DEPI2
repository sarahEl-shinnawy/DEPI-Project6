import os
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ============================================================
# 1. Load Preprocessed Data
# ============================================================
pickle_path = "/content/drive/MyDrive/DEPI-Project6/DEPI-Project6/data_augmented.pickle"

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

print("Keys in pickle:", data.keys())

# Extract features and labels
X = np.array(data['data'])
y = np.array(data['labels'])

print("Images shape:", X.shape)
print("Labels shape:", y.shape)

# ============================================================
# 2. Encode Labels
# ============================================================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_int = le.fit_transform(y)

# Ensure save directory exists
save_dir = "/content/drive/MyDrive/SignLanguageProject"
os.makedirs(save_dir, exist_ok=True)

# Save label encoder
joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))
print("Label encoder saved at:", os.path.join(save_dir, "label_encoder.pkl"))

# One-hot encoding
y_encoded = to_categorical(y_int, num_classes=len(np.unique(y_int)))
print("One-hot encoded labels shape:", y_encoded.shape)

# ============================================================
# 3. Split Dataset
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_int
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# ============================================================
# 4. Build Model
# ============================================================
model = Sequential([
    Dense(256, activation='relu', input_shape=(42,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(len(np.unique(y_int)), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 5. Prevent Overfitting (Callbacks)
# ============================================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# ============================================================
# 6. Train the Model
# ============================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, reduce_lr]
)

# ============================================================
# 7. Evaluate Model
# ============================================================
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ============================================================
# 8. Save Model
# ============================================================
model.save("/content/drive/MyDrive/SignLanguageProject/En_model2.h5")
print("Model saved successfully!")

# Save with pickle (not recommended, but included)
with open("/content/drive/MyDrive/SignLanguageProject/En_model2.p", "wb") as f:
    pickle.dump(model, f)
print("Model saved with pickle!")

# ============================================================
# 9. Load Model
# ============================================================
model = load_model("/content/drive/MyDrive/SignLanguageProject/En_model2.h5")
print("Model loaded successfully!")

# ============================================================
# 10. Predictions
# ============================================================
# Predict on a single sample
sample = X_test[0:1]
prediction = model.predict(sample)
predicted_class = np.argmax(prediction)
predicted_letter = le.inverse_transform([predicted_class])[0]
true_letter = le.inverse_transform([np.argmax(y_test[0])])[0]

print("Predicted letter:", predicted_letter)
print("True letter:", true_letter)

# Predict on multiple samples
for i in range(10):
    sample = X_test[i:i+1]
    pred_class = np.argmax(model.predict(sample))
    pred_letter = le.inverse_transform([pred_class])[0]
    true_letter = le.inverse_transform([np.argmax(y_test[i])])[0]
    print(f"Predicted: {pred_letter}, True: {true_letter}")

# ============================================================
# 11. Visualization
# ============================================================
epochs_ran = range(1, len(history.history['accuracy']) + 1)

# Accuracy Plot
plt.plot(epochs_ran, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs_ran, history.history['val_accuracy'], label='Validation Accuracy')
plt.axvline(len(epochs_ran), color='red', linestyle='--', label='Stopped Epoch')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss Plot
plt.plot(epochs_ran, history.history['loss'], label='Train Loss')
plt.plot(epochs_ran, history.history['val_loss'], label='Validation Loss')
plt.axvline(len(epochs_ran), color='red', linestyle='--', label='Stopped Epoch')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
