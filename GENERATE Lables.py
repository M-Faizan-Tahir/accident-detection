import json
import pandas as pd
import os

def extract_labels(coco_json_path, image_dir):
    # Load COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get images and annotations
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # Create a dictionary to store image labels
    image_labels = {img['id']: {'file_name': img['file_name'], 'label': 0} for img in images}
    
    # Mark images with category_id: 1 as Accident (1)
    for ann in annotations:
        if ann['category_id'] == 1:
            image_labels[ann['image_id']]['label'] = 1
    
    # Convert to DataFrame
    labels_df = pd.DataFrame([
        {'image': data['file_name'], 'label': data['label']}
        for img_id, data in image_labels.items()
    ])
    
    # Verify image files exist
    labels_df['image_path'] = labels_df['image'].apply(lambda x: os.path.join(image_dir, x))
    labels_df = labels_df[labels_df['image_path'].apply(os.path.exists)]
    
    return labels_df

# Usage for validation set
valid_json_path = 'DATASET/valid/_annotations.coco.json'
valid_image_dir = 'DATASET/valid/'
valid_labels = extract_labels(valid_json_path, valid_image_dir)
print(valid_labels.head())
valid_labels.to_csv('valid_labels.csv', index=False)

# Usage for train set
train_json_path = 'DATASET/train/_annotations.coco.json'
train_image_dir = 'DATASET/train/'
train_labels = extract_labels(train_json_path, train_image_dir)
print(train_labels.head())
train_labels.to_csv('train_labels.csv', index=False)

# Usage for test set
test_json_path = 'DATASET/test/_annotations.coco.json'
test_image_dir = 'DATASET/test/'
test_labels = extract_labels(test_json_path, test_image_dir)
print(test_labels.head())
test_labels.to_csv('test_labels.csv', index=False)