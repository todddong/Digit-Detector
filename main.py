import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import center_of_mass

# mnist = tf.keras.datasets.mnist

# # load the data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # normalize the data
# # x_train = tf.keras.utils.normalize(x_train, axis = 1)
# # x_test = tf.keras.utils.normalize(x_test, axis = 1)

# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# # model.compile(optimizer=tf.keras.optimizers.Adam(0.00001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(x_train, y_train, epochs = 5)

# model.save('number_recognition.keras')

# # Load
model = tf.keras.models.load_model('number_recognition.keras')

# loss, accuracy = model.evaluate(x_test, y_test)

# # print(loss)
# # print(accuracy)

def load_image_robust(filename):
    try:
        import matplotlib.image as mpimg
        img_plt = mpimg.imread(filename)
        
        # If there are colors make it grayscale
        if len(img_plt.shape) == 3:
            img_plt = np.mean(img_plt, axis=2)
        
        # Convert to proper format
        if img_plt.dtype == np.float32 or img_plt.dtype == np.float64:
            img_plt = (img_plt * 255).astype(np.uint8)
        
        return img_plt

    except Exception as e:
        print(f"File failed: {e}")
        return None

def preprocess_image(img):
    # Resize to 28x28 (MNIST format)
    img_resized = cv2.resize(img, (28, 28))
    
    # Check if we need to invert colors
    mean_pixel = np.mean(img_resized)
    print(f"Mean pixel value: {mean_pixel:.2f}")
    
    if mean_pixel > 127:  # Black digit on white background
        print("Inverting colors...")
        img_processed = 255 - img_resized
    else:
        print("Using original colors...")
        img_processed = img_resized
    
    # Normalize to 0–1
    img_processed = img_processed.astype(np.float32) / 255.0
    
    # Center the digit using center of mass
    cy, cx = center_of_mass(img_processed)
    shift_x = np.round(14 - cx).astype(int)
    shift_y = np.round(14 - cy).astype(int)
    
    # Apply shifts with proper boundary handling
    img_centered = np.roll(img_processed, shift_y, axis=0)
    img_centered = np.roll(img_centered, shift_x, axis=1)
    
    return img_resized, img_processed, img_centered

def predict_digit(model, img_centered):
    # Reshape for prediction - add batch dimension
    img_input = img_centered.reshape(1, 28, 28)
    
    # Predict
    prediction = model.predict(img_input, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return prediction, predicted_digit, confidence, img_input

def display_results(img_original, img_processed, img_input, prediction, predicted_digit, confidence):
    # Show images
    plt.figure(figsize=(12, 4))
    
    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Processed
    plt.subplot(1, 3, 2)
    plt.imshow(img_processed, cmap='gray')
    plt.title("Processed (28x28)")
    plt.axis('off')
    
    # Final input to model
    plt.subplot(1, 3, 3)
    plt.imshow(img_input[0], cmap='gray')
    plt.title("Input to Model (Centered)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Output results
    print(f"\n Prediction Results:")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show all probabilities
    print(f"\n📊 All probabilities:")
    for i in range(10):
        prob = prediction[0][i] * 100

# Main execution
try:
    # Process images digit0.png through digit8.png
    for num in range(13):
        print(f"\n{'='*50}")
        print(f"Processing digit{num}.png")
        print(f"{'='*50}")
        
        # Load image
        img = load_image_robust(f'digit{num}.png')
        
        if img is None:
            print("❌ Could not load file")
            continue
        
        print(f"Original image shape: {img.shape}")
        
        # Preprocess image
        img_resized, img_processed, img_centered = preprocess_image(img)
        
        # Make prediction
        prediction, predicted_digit, confidence, img_input = predict_digit(model, img_centered)
        
        # Display results
        display_results(img, img_processed, img_input, prediction, predicted_digit, confidence)

except Exception as e:
    print(f"❌ Error processing image: {str(e)}")
    import traceback
    traceback.print_exc()