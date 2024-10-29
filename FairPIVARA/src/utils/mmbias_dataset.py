import os
from PIL import Image
from torch.utils.data import Dataset

class MMBiasDataset(Dataset):
    def __init__(self, root_dir, image_preprocessor):
        self.root_dir = root_dir
        self.image_path = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith('.jpg') and filename != '.DS_Store.jpg']
        self.transform = image_preprocessor

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        return image