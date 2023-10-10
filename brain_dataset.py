import numpy as np
import random as random
import cv2
import torch
import albumentations as A
from torch.utils.data import Dataset
import torchvision.transforms as tt

# This BrainDataset class is designed to be used with PyTorch's DataLoader 
# to efficiently load and preprocess data for training, validation, or testing in a deep learning pipeline. 
class BrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0])
        image = np.array(image) / 255.
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        mask = np.array(mask) / 255.
        
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.float32)
        image = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))
        mask = torch.from_numpy(mask).type(torch.float32)
        
        return image, mask

# This function is used to print out the size of the dataset and the size of a sample image and mask
def dataset_info(dataset):
    print(f'Size of dataset: {len(dataset)}')
    index = random.randint(1, 40)
    img, label = dataset[index]
    print(f'Sample-{index} Image size: {img.shape}, Mask: {label.shape}\n')

# This function is used to set the seed for reproducibility
def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()

def print_model_architecture(model):
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    


if __name__ == "__main__":
    pass
