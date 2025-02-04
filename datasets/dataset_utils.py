import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageListDataset(Dataset):
    """
    Generic image loading Dataset:
    - Accepts a list: [(img_path, label), (img_path, label), ...]
    - label=-1 indicates that the target domain has no labels
    - transform: Preprocessing/augmentation applied to images
    """
    def __init__(self, img_paths_labels, transform=None):
        self.img_paths_labels = img_paths_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths_labels)

    def __getitem__(self, idx):
        path, label = self.img_paths_labels[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_office31_filelist(domain_root):
    """
    Retrieves a list of (image_path, label_id) from a specific domain folder in Office-31.
    Assumptions:
      - Each subfolder under domain_root corresponds to a category.
    Returns:
      [(img_path, label_id), ...] which may be an empty list (if the path does not exist or the folder is empty).
    """
    image_list = []
    if not os.path.isdir(domain_root):
        print(f"[Warning] domain_root={domain_root} does not exist or is not a directory, returning an empty list.")
        return image_list

    # Filter subdirectories (category names)
    class_names = sorted([d for d in os.listdir(domain_root)
                          if os.path.isdir(os.path.join(domain_root, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    for cls_name in class_names:
        cls_dir = os.path.join(domain_root, cls_name)
        for f in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, f)
            if os.path.isfile(fpath):
                label_id = class_to_idx[cls_name]
                image_list.append((fpath, label_id))

    return image_list


def get_office31_dataloaders(
    source_root, 
    target_root, 
    batch_size=32, 
    resize_size=256,
    crop_size=224,
    random_seed=2023,
    for_test=False
):
    """
    Constructs DataLoaders for the source/target domains of Office-31.
    - source_root: Source domain folder (e.g., ./data/Office31/amazon)
    - target_root: Target domain folder (e.g., ./data/Office31/webcam)
    - batch_size: Batch size
    - resize_size: Resize images to this size first
    - crop_size: Random/center crop size
    - random_seed: Random seed
    - for_test: If True, use test-time augmentation for the target domain (otherwise, use training augmentation).
    Returns:
      source_loader, target_loader
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Training augmentations for the source domain
    transform_train_source = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Choose augmentation strategy for the target domain based on for_test flag
    if not for_test:
        transform_target = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_target = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # Retrieve file lists
    source_list = get_office31_filelist(source_root)
    target_list_raw = get_office31_filelist(target_root)

    # Target domain has no labels => label=-1
    target_list = [(p, -1) for (p, lbl) in target_list_raw]

    # Create Dataset instances
    source_ds = ImageListDataset(source_list, transform=transform_train_source)
    target_ds = ImageListDataset(target_list, transform=transform_target)

    # Create DataLoader instances
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, drop_last=True) if len(source_ds) > 0 else None
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True, drop_last=True) if len(target_ds) > 0 else None

    return source_loader, target_loader
