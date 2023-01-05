import tensorflow as tf
import numpy as np
import glob

model = tf.keras.models.load_model('model.h5')

img_paths = glob.glob('./predict_images/*.jpg')

# img_paths = ['predict_images/pred0.jpg', 'predict_images/pred1.jpg',
#              'predict_images/pred2.jpg', 'predict_images/pred3.jpg']

images = [tf.keras.preprocessing.image.load_img(
    img_path, target_size=(150, 150)) for img_path in img_paths]

images = [tf.keras.preprocessing.image.img_to_array(image) for image in images]

images = np.array(images)

predictions = model.predict(images)
predictions = (predictions > 0.5).astype(int)

for i in range(len(predictions)):
    print(
        f'{img_paths[i]} is {"not" if not predictions[i][0] else ""} a crosswalk sign')
