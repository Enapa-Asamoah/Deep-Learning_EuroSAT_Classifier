import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class TrashNetDataset(Dataset):
    """
    PyTorch Dataset for TrashNet
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str or Path): Root directory with class subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_folder = self.root_dir / cls
            for img_file in cls_folder.glob('*.*'):
                self.samples.append((img_file, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
