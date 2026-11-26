"""
ImageNet100 Dataset Download Script

This script downloads and prepares the ImageNet100 dataset.
ImageNet100 is a subset of ImageNet containing 100 classes.

Sources:
1. Kaggle (recommended): https://www.kaggle.com/datasets/ambityga/imagenet100
2. Official ImageNet website (requires registration): https://image-net.org/
3. Common academic mirrors and preprocessed versions
"""

import os
import argparse
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download ImageNet100 Dataset')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store the dataset')
    parser.add_argument('--source', type=str, default='kaggle',
                        choices=['kaggle', 'manual'],
                        help='Download source')
    return parser.parse_args()


def download_from_kaggle(data_dir):
    """Download ImageNet100 from Kaggle using kagglehub"""
    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub is not installed.")
        print("Please install it with: pip install kagglehub")
        return False
    
    print("Downloading ImageNet100 from Kaggle...")
    print("Dataset: ambityga/imagenet100")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("ambityga/imagenet100")
        print(f"Downloaded to: {path}")
        
        # Move to target directory
        imagenet100_dir = os.path.join(data_dir, 'imagenet100')
        
        if os.path.exists(imagenet100_dir):
            print(f"Directory {imagenet100_dir} already exists. Removing...")
            shutil.rmtree(imagenet100_dir)
        
        # Check the downloaded structure
        if os.path.isdir(path):
            # If downloaded as a directory, check its contents
            contents = os.listdir(path)
            
            # Look for train and val folders
            if 'train' in contents and 'val' in contents:
                # Already has correct structure
                shutil.move(path, imagenet100_dir)
            else:
                # Create imagenet100 folder and move contents
                os.makedirs(imagenet100_dir, exist_ok=True)
                for item in contents:
                    shutil.move(os.path.join(path, item), imagenet100_dir)
        else:
            # If it's a file (zip), extract it
            print("Extracting downloaded files...")
            extract_archive(path, data_dir)
            os.remove(path)
        
        print(f"\nImageNet100 dataset ready at: {imagenet100_dir}")
        return True
        
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("\nPlease ensure you have:")
        print("1. Installed kagglehub: pip install kagglehub")
        print("2. Configured Kaggle API credentials")
        print("   - Get API token from: https://www.kaggle.com/settings/account")
        print("   - Place kaggle.json in ~/.kaggle/")
        return False


def download_file(url, output_path, desc="Downloading"):
    """Download file with progress"""
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r{desc}: {percent}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print()  # New line after progress


def extract_archive(archive_path, extract_to):
    """Extract tar.gz or zip archive"""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_to)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print("Extraction complete!")


def setup_imagenet100(data_dir):
    """
    Setup ImageNet100 dataset directory structure
    
    Expected structure after setup:
    data_dir/
    └── imagenet100/
        ├── train/
        │   ├── n02869837/
        │   ├── n01749939/
        │   └── ... (100 class folders)
        └── val/
            ├── n02869837/
            ├── n01749939/
            └── ... (100 class folders)
    """
    imagenet100_dir = os.path.join(data_dir, 'imagenet100')
    train_dir = os.path.join(imagenet100_dir, 'train')
    val_dir = os.path.join(imagenet100_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    return imagenet100_dir, train_dir, val_dir


def print_manual_instructions():
    """Print instructions for manual download"""
    print("""
================================================================================
ImageNet100 Dataset Download Instructions
================================================================================

ImageNet100 is a subset of ImageNet with 100 classes.

Option 1: Download from Kaggle (Recommended)
--------------------------------------------
Dataset: https://www.kaggle.com/datasets/ambityga/imagenet100

Steps:
1. Install kagglehub:
   pip install kagglehub

2. Get Kaggle API credentials:
   - Go to: https://www.kaggle.com/settings/account
   - Click "Create New Token"
   - Download kaggle.json
   - Place it in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)

3. Run this script:
   python download_imagenet100.py --source kaggle

Or use Python directly:
   import kagglehub
   path = kagglehub.dataset_download("ambityga/imagenet100")

Option 2: Download from Official ImageNet Website
-------------------------------------------------
1. Register at: https://image-net.org/download.php
2. Download ILSVRC2012 (ImageNet) dataset
3. Use the provided class list to extract 100 classes

Option 3: Use Academic/Research Mirrors
---------------------------------------
Some universities provide mirrors:
- Check if your institution has a local mirror
- Common mirrors: Academic torrents, research group shares

Option 4: Create from Full ImageNet
-----------------------------------
If you have full ImageNet, you can create ImageNet100:

The 100 classes in ImageNet100 are typically:
- First 100 classes from ImageNet (sorted by WordNet ID)
- Or a specific subset used in particular papers

Common ImageNet100 class list (first 10 shown):
- n02869837 (basketball)
- n01749939 (green mamba)
- n02488291 (langur)
- n02107142 (Doberman)
- n13037406 (guacamole)
- n02091831 (saluki)
- n04517823 (vacuum)
- n04589890 (washer)
- n03062245 (cocktail shaker)
- n01773797 (garden spider)
... (90 more classes)

================================================================================
Directory Structure
================================================================================

After downloading, organize your data as follows:

{data_dir}/
└── imagenet100/
    ├── train/
    │   ├── n02869837/
    │   │   ├── img1.JPEG
    │   │   └── ...
    │   ├── n01749939/
    │   │   └── ...
    │   └── ... (100 class folders)
    └── val/
        ├── n02869837/
        ├── n01749939/
        └── ... (100 class folders)

================================================================================
Alternative: Use Tiny ImageNet for Testing
================================================================================

For quick testing, you can use Tiny ImageNet (200 classes, 64x64 images):

Download link: http://cs231n.stanford.edu/tiny-imagenet-200.zip

Or use this command:
    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    unzip tiny-imagenet-200.zip

Note: Tiny ImageNet has different class IDs than ImageNet100.

================================================================================
""")


def check_existing_dataset(data_dir):
    """Check if ImageNet100 dataset already exists"""
    imagenet100_dir = os.path.join(data_dir, 'imagenet100')
    train_dir = os.path.join(imagenet100_dir, 'train')
    val_dir = os.path.join(imagenet100_dir, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_classes = [d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))]
        val_classes = [d for d in os.listdir(val_dir) 
                      if os.path.isdir(os.path.join(val_dir, d))]
        
        if len(train_classes) > 0 and len(val_classes) > 0:
            print(f"Found existing ImageNet100 dataset:")
            print(f"  Train classes: {len(train_classes)}")
            print(f"  Val classes: {len(val_classes)}")
            return True
    
    return False


def main():
    args = parse_args()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Check if dataset already exists
    if check_existing_dataset(args.data_dir):
        print("\nDataset already exists! No need to download.")
        print(f"Location: {os.path.join(args.data_dir, 'imagenet100')}")
        return
    
    print("=" * 80)
    print("ImageNet100 Dataset Download")
    print("=" * 80)
    
    # Try Kaggle download first if selected
    if args.source == 'kaggle':
        success = download_from_kaggle(args.data_dir)
        if success:
            print("\n" + "=" * 80)
            print("Download Complete!")
            print("=" * 80)
            print(f"\nYou can now run training with:")
            print(f"   python train.py --data-root {os.path.join(args.data_dir, 'imagenet100')}")
            return
        else:
            print("\nKaggle download failed. Showing manual instructions...")
    
    # Setup directory structure for manual download
    imagenet100_dir, train_dir, val_dir = setup_imagenet100(args.data_dir)
    
    print(f"\nDataset will be stored in: {imagenet100_dir}")
    print("=" * 80)
    
    # Print manual instructions
    print_manual_instructions()
    
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print(f"""
1. Download ImageNet100 images from one of the sources above
2. Extract them to: {imagenet100_dir}
3. Ensure the directory structure matches:
   - {train_dir}/<class_id>/images...
   - {val_dir}/<class_id>/images...

4. Run training with:
   python train.py --data-root {imagenet100_dir}
""")


if __name__ == '__main__':
    main()
