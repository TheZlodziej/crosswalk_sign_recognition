import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('model.h5')

# Create an empty list to store the predictions
predictions = []

# Loop over the input images
input_imgs = ['predict_images/pred0.jpg', 'predict_images/pred1.jpg', 'predict_images/pred2.jpg']
for image_path in input_imgs:
  # Read the image file and resize it
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))

  # Convert the image to a NumPy array
  input_array = tf.keras.preprocessing.image.img_to_array(img)

  # Add an additional dimension to the array (since the model expects a batch of images, not a single image)
  input_array = np.expand_dims(input_array, axis=0)

  # Use the model to make a prediction
  prediction = model.predict(input_array)

  # Append the prediction to the list
  predictions.append(prediction)

# Print results
result = []
for i, entry in enumerate(predictions):
    # od 80% w gore
    if entry[0][0] + 0.2 >= 1:
        result.append(f"{input_imgs[i]} is a crosswalk sign")
    else:
        result.append(f"{input_imgs[i]} is not a crosswalk sign")

print("\n".join(result))