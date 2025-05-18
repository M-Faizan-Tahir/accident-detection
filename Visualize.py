import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 1. Load the trained model
model_path = 'accident_detection_cnn.h5'  # Update with your model path
model = load_model(model_path)
print("Model loaded successfully.")

# 2. Reinitialize test_labels from CSV
test_labels = pd.read_csv('test_labels.csv')  # Update with your CSV path
test_labels['label'] = test_labels['label'].astype(str)
print(test_labels.head())
print(f"Total images: {len(test_labels)}")
print(f"Accident images: {sum(test_labels['label'] == 1)}")
print(f"Non-Accident images: {sum(test_labels['label'] == 0)}")

# 3. Initialize test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_labels,
    directory=None,  # Use full paths from image_path
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),  # Match training input size
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 4. Generate predictions
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()  # Binary classification threshold
true_classes = test_generator.labels

# 5. Print predictions
for img_name, pred, true in zip(test_generator.filenames, predicted_classes, true_classes):
    print(f'Image: {img_name}, Predicted: {"Accident" if pred else "Non-Accident"}, True: {"Accident" if true else "Non-Accident"}')

# 6. Create and save confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Accident', 'Accident'])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Accident Detection Model')
plt.tight_layout()

# Save the plot as an image
output_image_path = 'confusion_matrix.png'  # Update with desired path
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Confusion matrix saved as {output_image_path}")

# 7. Optional: Print classification metrics
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=['Non-Accident', 'Accident']))