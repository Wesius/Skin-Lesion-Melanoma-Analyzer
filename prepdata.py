import os
import csv
import random
import shutil
import numpy as np
from tqdm import tqdm
from collections import Counter
import albumentations as A
from PIL import Image


def load_data(csv_path, image_dir):
    images, labels = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading data"):
            img_path = os.path.join(image_dir, row['image'] + '.jpg')
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            label = 1 if row['MEL'] == '1.0' else 0  # 1 for melanoma, 0 for non-melanoma
            images.append(img_path)
            labels.append(label)
    return images, labels


def split_data(images, labels):
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    train_split = int(0.7 * len(images))
    val_split = int(0.9 * len(images))

    train_images, train_labels = images[:train_split], labels[:train_split]
    val_images, val_labels = images[train_split:val_split], labels[train_split:val_split]
    test_images, test_labels = images[val_split:], labels[val_split:]

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def augment_melanoma_images(images, labels, target_count):
    melanoma_indices = [i for i, label in enumerate(labels) if label == 1]

    augmented_images = list(images)
    augmented_labels = list(labels)

    augmentation = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.ElasticTransform(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.HueSaturationValue(10, 15, 10),
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),
        ], p=0.3),
    ])

    current_melanoma_count = len(melanoma_indices)
    while current_melanoma_count < target_count:
        idx = random.choice(melanoma_indices)
        img_path = images[idx]
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)

        augmented = augmentation(image=image_np)
        augmented_image = Image.fromarray(augmented['image'])

        base_name = os.path.basename(img_path)
        new_name = f"mel_aug_{current_melanoma_count}_{base_name}"
        new_path = os.path.join(os.path.dirname(img_path), new_name)
        augmented_image.save(new_path)

        augmented_images.append(new_path)
        augmented_labels.append(1)
        current_melanoma_count += 1

    return augmented_images, augmented_labels


def balance_dataset(images, labels):
    melanoma_count = sum(labels)
    non_melanoma_count = len(labels) - melanoma_count
    target_count = max(melanoma_count, non_melanoma_count)

    if melanoma_count < target_count:
        images, labels = augment_melanoma_images(images, labels, target_count)

    return images, labels


def main():
    csv_path = 'GroundTruth.csv'
    image_dir = 'images'
    output_base_dir = 'prepared_dataset'

    # Load data
    images, labels = load_data(csv_path, image_dir)
    print(f"Total images loaded: {len(images)}")

    # Split data
    train_data, val_data, test_data = split_data(images, labels)

    # Balance and prepare each dataset
    for name, (imgs, lbls) in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        print(f"\nPreparing {name} dataset:")
        print(f"Original distribution: {Counter(lbls)}")

        balanced_imgs, balanced_lbls = balance_dataset(list(imgs), list(lbls))
        print(f"Balanced distribution: {Counter(balanced_lbls)}")

        output_dir = os.path.join(output_base_dir, name)
        prepare_dataset(balanced_imgs, balanced_lbls, output_dir)

        final_distribution = Counter([1 if 'mel_' in img else 0 for img in os.listdir(output_dir)])
        print(f"Final {name} dataset distribution: {final_distribution}")

def prepare_dataset(images, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_path, label in zip(images, labels):
        base_name = os.path.basename(img_path)
        prefix = "mel_" if label == 1 else "nomel_"
        new_name = prefix + base_name
        new_path = os.path.join(output_dir, new_name)
        shutil.copy(img_path, new_path)


if __name__ == "__main__":
    main()