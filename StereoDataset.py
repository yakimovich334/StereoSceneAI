from torch.utils.data import Dataset
from PIL import Image
import os

class StereoDataset(Dataset):
    def __init__(self, left_dir, right_dir, transform=None):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.transform = transform
        self.left_images = sorted(os.listdir(left_dir))
        self.right_images = sorted(os.listdir(right_dir))

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img_path = os.path.join(self.left_dir, self.left_images[idx])
        right_img_path = os.path.join(self.right_dir, self.right_images[idx])

        left_img = Image.open(left_img_path).convert('RGB')
        right_img = Image.open(right_img_path).convert('RGB')

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        return left_img, right_img
