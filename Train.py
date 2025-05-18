from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from PIL import Image

train_labels = pd.read_csv('train_labels.csv')
valid_labels = pd.read_csv('valid_labels.csv')
test_labels = pd.read_csv('test_labels.csv')

train_labels['label'] = train_labels['label'].astype(str)
valid_labels['label'] = valid_labels['label'].astype(str)
test_labels['label'] = test_labels['label'].astype(str)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_labels,
    directory='DATASET/train',
    x_col='image',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_labels,
    directory='DATASET/valid',
    x_col='image',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=valid_generator,
    verbose=1
)
model.save('accident_detection_cnn.h5')
model.save('accident_detection_cnn.keras')



test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_labels,
    directory='DATASET/test',
    x_col='image',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')