import glob
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
from monai import transforms

import traceback

class MRI_dataset(Dataset):
    def __init__(self, root_dir='./data', transform=None) -> None:
        super().__init__()
        assert root_dir

        self.scan_IDs = glob.glob(f'{root_dir}/*')
        self.transform = transform
        self.flip = transforms.RandFlipd(keys=['label'], prob=1, spatial_axis=[0])

    def __getitem__(self, index):
        return self.get_SCANS(self.scan_IDs[index])

    def __len__(self):
        return len(self.scan_IDs)
    
    def get_SCANS(self, path):
        cond = torch.load(f'{path}/sex-age-ventV-brainV.pt')
        
        prob =  np.random.random()
        latent = torch.load(f'{path}/latent_f.pt')  if prob > 0.5 else torch.load(f'{path}/latent.pt')
        d = {
            'new':torch.load(f'{path}/new_semantics.pt'),
        }
        
        z_mu, z_sigma = latent['mu'], latent['sigma']
        
        try:
            if self.transform:
                d = self.transform(d)
                if prob > 0.5:
                    d = self.flip(d)
                
        except Exception as ex:
            print(path)
            print(ex)
            print(traceback.format_exc())
        
        return z_mu, z_sigma, cond, d['label'], path
    

if __name__ == "__main__":
    # from utils.transform_generator import MONAI_transformerd
    
    from monai import transforms
    
    transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image",'label']),
        transforms.EnsureChannelFirstd(keys=["image",'label'], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
        transforms.Orientationd(keys=["image", 'label'], axcodes="RAS"), # torch.Size([240, 240, 155])
        transforms.CenterSpatialCropd(keys=["image", 'label'], roi_size=(196, 224, 224)),
        transforms.Lambdad(keys=["image", 'label'], func=lambda x: x[:, :, :, 60:]),
        transforms.CenterSpatialCropd(keys=["image", 'label'], roi_size=(160, 224, 160)),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.AsDiscreted(keys=['label'], to_onehot=47),
        transforms.Resized(keys=['label'], spatial_size=(160, 224, 160), mode='nearest'),
    ] 
)
    dataset = MRI_dataset(transform=transforms)
    
    data = dataset[0]
    print(data['image'].shape, data['label'].shape, data['cond'].shape)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, persistent_workers=True)
    
    

