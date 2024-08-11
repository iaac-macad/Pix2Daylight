import shutil
import random
from pathlib import Path


def split_dataset(source_dir: str, dest_dir: str, train_ratio: float, seed: int = 42):
    """
    Split dataset into train, validation, and test sets.

    Parameters:
    source_dir (str): Directory with the original images.
    dest_dir (str): Directory where the split folders will be created.
    train_ratio (float): Ratio of training data.
    seed (int): Random seed for reproducibility.
    """
    


    
    train_dir = Path(dest_dir) / 'train'
    test_dir = Path(dest_dir) / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # List all image files
    image_files = [f for f in Path(source_dir).iterdir() if f.is_file()]


    random.seed(seed)
    random.shuffle(image_files)

  
    total_images = len(image_files)
    train_end = int(train_ratio * total_images)

    train_files = image_files[:train_end]
    test_files = image_files[train_end:]

  
    for file in train_files:
        shutil.copy(file, train_dir / file.name)
    for file in test_files:
        shutil.copy(file, test_dir / file.name)


    print(f"Total images: {total_images}")
    print(f"Training images: {len(train_files)}")
    print(f"Test images: {len(test_files)}")


