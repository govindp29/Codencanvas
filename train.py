import tensorflow as tf
import numpy as np

# 1. CREATE DATA (R, G, B values)
# We teach it 3 colors: Red, Green, Blue
# Format: [Red, Green, Blue] -> Label (0=Red, 1=Green, 2=Blue)
inputs = np.array([
    [255, 0, 0],   [240, 10, 10], [200, 50, 50], # Examples of Red
    [0, 255, 0],   [10, 240, 10], [50, 200, 50], # Examples of Green
    [0, 0, 255],   [10, 10, 240], [50, 50, 200], # Examples of Blue
], dtype=float) / 255.0  # Normalize to 0-1 range

outputs = np.array([
    0, 0, 0,  # Red labels
    1, 1, 1,  # Green labels
    2, 2, 2   # Blue labels
])

# 2. BUILD MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)), # 3 inputs (R,G,B)
    tf.keras.layers.Dense(3, activation='softmax') # 3 outputs (Probabilities for Red, Green, Blue)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. TRAIN
print("Training model...")
model.fit(inputs, outputs, epochs=500, verbose=0)

# 4. CONVERT TO TFLITE (The "Tiny" format)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 5. SAVE FILE
with open('color_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! Saved 'color_model.tflite'")