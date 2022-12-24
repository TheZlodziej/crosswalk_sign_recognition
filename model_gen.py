import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Preprocessing function
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [150, 150])
  image /= 255.0  # normalize to [0,1] range
  return image

# Load images and labels
def load_dataset(image_dir):
  labels = []
  images = []
  class_names = os.listdir(image_dir)
  for index, class_name in enumerate(class_names):
    for image_path in os.listdir(os.path.join(image_dir, class_name)):
      image_raw = tf.io.read_file(os.path.join(image_dir, class_name, image_path))
      image = preprocess_image(image_raw)
      images.append(image)
      labels.append(index)
  return np.array(images), np.array(labels), class_names

# Load the dataset
image_dir = './images'
X, y, class_names = load_dataset(image_dir)

# Split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Create a TensorFlow model using the Keras API
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)

# Save in model/ subfolder
model.save('model.h5')