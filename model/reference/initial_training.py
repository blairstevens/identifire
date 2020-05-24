import tensorflow as tf
import glob
import matplotlib.pyplot as plt

image_root = ('data/*')

for item in glob.glob(image_root):
    print(item)

list_ds = tf.data.Dataset.list_files('data/*/*')

# def process_path(file_path):
#   label = tf.strings.split(file_path, '/')[-2]
#   return tf.io.read_file(file_path), label
#
# labeled_ds = list_ds.map(process_path)

def parse_image(filename):
  parts = tf.strings.split(filename, '/')
  label = parts[-2]

  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [128, 128])
  return image, label

file_path = next(iter(list_ds))
image, label = parse_image(file_path)

def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')

show(image, label)

images_ds = list_ds.map(parse_image)

for image, label in images_ds.take(2):
  show(image, label)

batch_size = 32
IMAGE_SIZE =128

image.shape

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

images_ds
