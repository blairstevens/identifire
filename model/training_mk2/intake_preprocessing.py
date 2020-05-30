import tensorflow as tf
import glob
import numpy as np
import pathlib
import IPython.display as display
from PIL import Image

#point to your images
img_dir = ('model/training_mk2/set_data/')



data_dir = pathlib.Path(img_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
image_count

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])
CLASS_NAMES

ak = list(data_dir.glob('ak 47/*'))

for image_path in ak[:3]:
    display.display(Image.open(str(image_path)))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

for f in list_ds.take(4):
  print(f.numpy())

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES










def get_images_paths(img_path):
    list_img = tf.data.Dataset.list_files(img_path)
    paths = next(iter(list_img))
    return paths

def parse_images(paths):
    parts = tf.string.split(paths, '/')
    label = parts[-2]

    image = tf.io.read_file(paths)
