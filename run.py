import tensorflow as tf
import numpy as np

# 1. LOAD THE TINY BRAIN
# On a real Raspberry Pi, you would use 'tflite_runtime' instead of 'tensorflow' to save space.
interpreter = tf.lite.Interpreter(model_path="color_model.tflite")
interpreter.allocate_tensors()

# Get input and output details (so we know how to talk to the model)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("AI Color Classifier Ready! (Type 'exit' to quit)")
print("Enter RGB values (e.g., 255 0 0)")

while True:
    user_input = input("RGB > ")
    if user_input == 'exit': break
    
    try:
        # Parse user input "255 0 0" -> [255, 0, 0]
        r, g, b = map(int, user_input.split())
        
        # Normalize (divide by 255 like we did in training)
        input_data = np.array([[r, g, b]], dtype=np.float32) / 255.0
        
        # 2. SET INPUT
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # 3. RUN INFERENCE
        interpreter.invoke()
        
        # 4. GET OUTPUT
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data) # Find the highest probability
        
        # Translate number to text
        colors = ["Red", "Green", "Blue"]
        print(f"I think this is: {colors[prediction]} ({output_data[0][prediction]:.2%} sure)")
        
    except:
        print("Invalid input. Try format: 255 0 0")