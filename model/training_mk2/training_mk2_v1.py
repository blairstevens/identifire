import tensorflow as tf
import tensorflowjs as tfjs
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 160
BATCH_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SHUFFLE_BUFFER_SIZE = 1000

train_datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    'model/training_mk2/set_data_split/train/',
    target_size=(IMG_SIZE,IMG_SIZE),
    class_mode='categorical',
    batch_size = BATCH_SIZE)

validation_generator = validation_datagen.flow_from_directory(
    'model/training_mk2/set_data_split/validation/',
    target_size=(IMG_SIZE,IMG_SIZE),
    class_mode='categorical',
    batch_size = BATCH_SIZE)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

print(train_generator)
print(validation_generator)

images_batch, labels_batch = next(train_generator)
print(images_batch.dtype, images_batch.shape)
print(labels_batch.dtype, labels_batch.shape)

feature_batch = base_model(images_batch)
print(feature_batch.shape)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),     #RMSprop(lr=base_learning_rate)
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

len(model.trainable_variables)

initial_epochs = 10
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_generator, steps = validation_steps)

history = model.fit(train_generator,
                    epochs=initial_epochs,
                    validation_data=validation_generator)


history
model

model

!mkdir -p saved_model
model.save('saved_model/my_model')
