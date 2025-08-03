#Project executable Files
#Step 1: Importing required libraries
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# Step 2: Load dataset
import kagglehub
phucthaiv02_butterfly_image_classification_path = kagglehub.dataset_download('phucthaiv02/butterflyimage-classification')
print('Data source import complete.')
from google.colab import drive
import kagglehub
path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification"
Step 3: Load label file
labels_df = pd.read_csv('/content/Training_set.csv')
#printing the csv files
df
# Step 4: Image augmentation
train_datagen = ImageDataGenerator(
rescale=1./255, 
validation_split=0.2, 
rotation_range=10, 
width_shift_range=0.1, 
height_shift_range=0.1, 
zoom_range=0.1,
horizontal_flip=True,
fill_mode='nearest' )
train_generator = train_datagen.flow_from_dataframe(
dataframe=labels_df, 
directory=path + '/train', 
x_col='filename', 
y_col='label', 
subset='training', 
target_size=(128, 128), 
batch_size=32, class_mode='categorical' )
val_generator = train_datagen.flow_from_dataframe(
dataframe=labels_df,
directory=path + '/train',
x_col='filename',
y_col='label', 
subset='validation', 
target_size=(128, 128), 
batch_size=32, 
class_mode='categorical' )
Step 5: Model building
model = Sequential([
Conv2D(32, (3,3), 
activation='relu', 
input_shape=(128,128,3)),
BatchNormalization(),
MaxPooling2D(2,2), 
Conv2D(64, (3,3), 
activation='relu'), 
BatchNormalization(),
MaxPooling2D(2,2), 
Conv2D(128, (3,3), 
activation='relu'), 
BatchNormalization(), 
MaxPooling2D(2,2),
Flatten(), 
Dense(128,
activation='relu'), 
Dense(512, activation='relu'),
Dense(train_generator.num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Step 6: Training
history = model.fit(
train_generator, validation_data=val_generator, epochs=30
)
Step 7: Evaluation
lt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()
# Make predictions
val_images, val_labels = next(val_generator)
pred_labels = model.predict(val_images)
pred_labels = np.argmax(pred_labels, axis=1)
true_labels = np.argmax(val_labels, axis=1)
# Visualization
class_indices = val_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}
def display_images(images, true_labels, pred_labels, class_names, num_images=9):
plt.figure(figsize=(15, 8))
for i in range(num_images):
plt.subplot(3, 3, i + 1)
plt.imshow(images[i])
true_label = class_names[int(true_labels[i])]
pred_label = class_names[int(pred_labels[i])]
plt.title(f"True: {true_label}\nPred: {pred_label}")
plt.axis('off')
plt.tight_layout()
plt.show()
display_images(val_images, true_labels, pred_labels, class_names)
