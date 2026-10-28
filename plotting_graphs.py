import pandas as pd
import os
import matplotlib.pyplot as plt
from decode_masks import rle_to_mask

def run_eda(csv_path: str, images_dir: str, output_dir: str):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Image Sizes (Severstal dataset is fixed at 256x1600)
    print("Image sizes: All images are fixed at 256 x 1600 pixels.")

    # 2. Count Images Per Class
    # The provided value counts: 3: 5150, 1: 897, 4: 801, 2: 247
    class_counts = df['ClassId'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Defect Occurrences per Class')
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()

    # 3. Defect-Free Ratio
    total_images = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    images_with_defects = df['ImageId'].nunique()
    defect_free_images = total_images - images_with_defects
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([images_with_defects, defect_free_images], 
           labels=['Has Defect', 'Defect-Free'], 
           autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    ax.set_title('Defect-Free Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'defect_free_ratio.png'))
    plt.close()

    # 4. Mask Area Distribution
    # Apply the decoder to calculate the area (sum of 1s in the binary mask)
    df['MaskArea'] = df['EncodedPixels'].apply(lambda x: rle_to_mask(x).sum())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for class_id in range(1, 5):
        subset = df[df['ClassId'] == class_id]
        ax.hist(subset['MaskArea'], bins=50, alpha=0.5, label=f'Class {class_id}')
        
    ax.set_title('Mask Area Distribution by Class')
    ax.set_xlabel('Area (Pixels)')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mask_area_distribution.png'))
    plt.close()


run_eda('data/raw/severstal-steel-defect-detection/train.csv', 'data/raw/severstal-steel-defect-detection/train_images', './plots')