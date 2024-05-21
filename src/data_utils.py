from typing import Tuple, Dict
import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9

from data_transform import SimplicialPreTransfrom, prepare_data, qm9_to_ev
from modules.transforms.liftings.graph2simplicial.vietoris_rips_lift import SimplicialVietorisRipsLifting


def calc_mean_mad(loader: DataLoader) -> Tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    return mean, mad


def generate_loaders_qm9(dis: float, dim: int, target_name: str, batch_size: int, num_workers: int, debug = False) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if debug:
        data_root = f'./datasets/QM9_delta_{dis}_dim_{dim}_debug'
        dataset = QM9(root=data_root)
        print('About to prepare data')
        dataset = [prepare_data(graph, target_name, qm9_to_ev) for graph in tqdm(dataset, desc='Preparing data')]
        print('Data prepared')
        transform = SimplicialVietorisRipsLifting(complex_dim=dim, dis=dis)
        dataset = [transform(data) for data in dataset[:7]]
    else:
        data_root = f'./datasets/QM9_delta_{dis}_dim_{dim}'
        transform = SimplicialPreTransfrom(complex_dim=dim, dis=dis, target_name=target_name)
        dataset = QM9(root=data_root, pre_transform=transform)
        dataset = dataset.shuffle()

    # filter relevant index and update units to eV

    # train/val/test split
    if debug:
        n_train, n_test = 3, 5
    else:
        n_train, n_test = 100000, 110000

    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:n_test]
    val_dataset = dataset[n_test:]

    # dataloaders
    follow = [f"x_{i}" for i in range(dim+1)] + ['x']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, follow_batch=follow)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, follow_batch=follow)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, follow_batch=follow)

    return train_loader, val_loader, test_loader