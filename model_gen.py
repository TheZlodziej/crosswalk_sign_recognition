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
img_size = (150, 150)
batch_size = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
  './images/train',
  seed=420,
  validation_split=0.2,
  subset="training",
  image_size=img_size,
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  './images/train',
  seed=420,
  validation_split=0.2,
  subset="validation",
  image_size=img_size,
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  './images/test',
  seed=420,
  image_size=img_size,
  batch_size=batch_size)

# Create a TensorFlow model using the Keras API
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, epochs=10, validation_data=val_ds)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"test loss = {test_loss}\ntest accuracy={test_acc}")

# Save model
model.save('model.h5')