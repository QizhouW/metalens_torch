import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataset(Dataset):
    def __init__(self, dataset_args):
        self.dataset_args = dataset_args
        self.img_transforms = dataset_args['img_transforms']
        if self.img_transforms == {}:
            self.img_transforms = {
                'img_size': 224,
                'crop_size': 224,
                'intensity_threshold': 1.25,
                'random_rotation': 0,
                'disable_transforms': False,
            }

        self.avg_weight_bond = dataset_args['avg_weight_bond']
        self.disable_transforms = self.img_transforms['disable_transforms']
        self.intensity_threshold = self.img_transforms['intensity_threshold']
        self.y_norm_factor = dataset_args['y_norm_factor']

        if not self.disable_transforms:
            self.transform_ftns = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.img_transforms['img_size'], antialias=True),
                transforms.CenterCrop(self.img_transforms['crop_size'], ),
                transforms.RandomRotation(self.img_transforms['random_rotation']),
            ])

        data_dir = dataset_args['data_dir']
        csv_file = dataset_args['csv_file']
        self.avocado_frame = pd.read_csv(os.path.join(data_dir, csv_file), index_col=0)
        self.hsi_dir = os.path.join(data_dir, f'hsi{dataset_args["spectral_channels"]}')
        self.rgb_dir = os.path.join(data_dir, 'rgb')

        self._sample = []
        self._ID = []
        self._hsi = []
        self._rgb = []
        self._cxcy = []
        self._firmness = []
        self.load_data()

    def process_image(self, image):
        if self.disable_transforms:
            return image
        else:
            return self.transform_ftns(image)

    def load_data(self):
        for _, row in self.avocado_frame.iterrows():
            rgb_path = os.path.join(self.rgb_dir, f'{row.item_name}.png')
            hsi_path = os.path.join(self.hsi_dir, f'{row.item_name}.npy')
            rgb = Image.open(rgb_path).convert('RGB')
            hsi = np.load(hsi_path)
            hsi[hsi > self.intensity_threshold] = self.intensity_threshold
            self._hsi.append(hsi)
            self._rgb.append(rgb)
            self._ID.append(int(row['ID']))
            self._sample.append(row['item_name'])
            self._cxcy.append([row['cx'], row['cy']])
            firmness_values = row[4:13].values.astype('float')
            self._firmness.append(firmness_values)

    def __len__(self):
        return len(self._sample)

    def __getitem__(self, idx):
        """
        Returns a sample of the dataset
        :param idx: index of the sample
        this function shall be modified to fit the exact training models
        here is an example that outputs everyting in the dataset
        """
        firmness_values = self._firmness[idx]
        weights = np.random.uniform(self.avg_weight_bond[0], self.avg_weight_bond[1], size=9)
        if self.y_norm_factor == 1:
            weighted_firmness = np.sum(weights * firmness_values) / np.sum(weights)
        else:
            weighted_firmness = np.sum(weights * firmness_values) / np.sum(weights) / self.y_norm_factor
            weighted_firmness = np.clip(weighted_firmness, 0, 1)
        y = torch.tensor(weighted_firmness, dtype=torch.float32).unsqueeze(0)
        hsi = self.process_image(self._hsi[idx])
        rgb = self.process_image(self._rgb[idx])
        return self._sample[idx], rgb, hsi, y


class PointWiseDataset(BaseDataset):
    def __init__(self, dataset_args):
        super().__init__(dataset_args)
        assert self.dataset_args['image_sample_pts'] > 0, "image_sample_pts must be greater than 0"

    def __getitem__(self, idx):
        points = []
        radius = 40
        cx, cy = self._cxcy[idx]
        firmness_values = self._firmness[idx]

        for _ in range(self.dataset_args['image_sample_pts']):
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))  # Square root for uniform distribution
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            points.append(self._hsi[idx][int(y), int(x)])
        points = np.mean(points,axis=0)

        weights = np.random.uniform(self.avg_weight_bond[0], self.avg_weight_bond[1], size=9)
        if self.y_norm_factor == 1:
            weighted_firmness = np.sum(weights * firmness_values) / np.sum(weights)
        else:
            weighted_firmness = np.sum(weights * firmness_values) / np.sum(weights) / self.y_norm_factor
            weighted_firmness = np.clip(weighted_firmness, 0, 1)
        return self._sample[idx], torch.tensor(points, dtype=torch.float32), torch.tensor(weighted_firmness, dtype=torch.float32).unsqueeze(0)


class MyDataLoader(DataLoader):
    """
    Loader for my dataset, be reused for each dataset
    """

    def __init__(self, dataset,data_args, collate_fn=default_collate):
        self.data_args = data_args
        self.dataset = dataset
        self.shuffle = self.data_args['shuffle']
        self.n_sample = len(self.dataset)
        self.sampler, self.valid_sampler = self._split_sampler(self.data_args['validation_split'])
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.data_args['batch_size'],
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': self.data_args['num_workers'],
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_sample)
        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_sample, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_sample * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_sample = len(train_idx)
        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


if __name__ =="__main__":
    # Parameters
    csv_file = 'data.csv'
    data_dir = '/home/wjoe/projects/digilens/avocado/'
    y_norm_factor = 1
    img_transforms = {}
    dataset_args = {
        'data_dir': data_dir,
        'csv_file': csv_file,
        'y_norm_factor': y_norm_factor,
        'img_transforms': img_transforms,
        "image_sample_pts": 10,
        "spectral_channels": 64,
        "avg_weight_bond": [1, 1],
    }
    # Create the dataset
    avocado_dataset = PointWiseDataset(dataset_args)
    # Create the dataloader
    # dataloader = DataLoader(avocado_dataset, batch_size=4, shuffle=True, num_workers=4)
