from typing import Dict

from src.utils.data import LJSpeechDataset, LJSpeechCollator
from torch.utils.data import DataLoader, Subset

from numpy.random import shuffle


def get_dataloaders(data_config: Dict):
    dataloaders = {'train': None, 'val': None}

    batch_size = data_config.get('batch_size', 1)
    num_workers = data_config.get('num_workers', 1)

    if 'val_split' in data_config:
        raise NotImplementedError

    dataset = LJSpeechDataset()
    if 'limit' in data_config:
        idx = list(range(len(dataset)))
        shuffle(idx)
        idx = idx[:data_config['limit']]

        dataset = Subset(dataset, idx)

    dataloaders['train'] = DataLoader(dataset,
                                      batch_size=batch_size,
                                      collate_fn=LJSpeechCollator(),
                                      shuffle=True,
                                      num_workers=num_workers)

    return dataloaders