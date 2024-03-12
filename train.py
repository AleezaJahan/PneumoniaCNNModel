import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from imutils import paths
from tensorflow.keras.utils import plot_model

# Defining constants and variables
img_width, img_height = 128, 128
train_data_dir = "/Users/aleezajahan/Downloads/chest_xray/train"
validation_data_dir = "/Users/aleezajahan/Downloads/chest_xray/val"
test_data_dir = "/Users/aleezajahan/Downloads/chest_xray/test"
BS = 64
EPOCHS = 10

# Creating train, validation, and test data generators
TRAIN = len(list(paths.list_images(train_data_dir)))
VAL = len(list(paths.list_images(validation_data_dir)))
TEST = len(list(paths.list_images(test_data_dir)))

trainAug = ImageDataGenerator(rescale=1./255, fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1./255, fill_mode="nearest")

trainGen = trainAug.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=BS,
    class_mode='binary')

valGen = valAug.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=BS,
    class_mode='binary')

# Loading pre-trained model and adding custom layers
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
preds = Dense(1, activation="sigmoid")(x)  # Change to 1 unit for binary classification

model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:16]:
    layer.trainable = False
for layer in model.layers[16:]:
    layer.trainable = True

model.compile(loss="binary_crossentropy",
              optimizer=SGD(learning_rate=0.001, momentum=0.9),
              metrics=["accuracy"])

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

H = model.fit(
    trainGen,
    epochs=EPOCHS,
    steps_per_epoch=TRAIN // BS,
    validation_data=valGen,
    validation_steps=VAL // BS,
    callbacks=[early])

model.save('model.keras')

# Evaluating the model
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.jpg")
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
