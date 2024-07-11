import lightning as L
import time
import wandb
import torch_geometric.transforms as T
from tqdm import tqdm

from torch_geometric.datasets import QM9

from modules.transforms.liftings.graph2simplicial.vietoris_rips_lifting import SimplicialVietorisRipsLifting, InvariantSimplicialVietorisRipsLifting
from modules.transforms.liftings.graph2simplicial.alpha_complex_lifting import SimplicialAlphaComplexLifting
from src.data_transform import InputPreprocTransform, LabelPreprocTransform, filter_not_enough_simplices_alpha
from torch_geometric.loader import DataLoader

LIFT_TYPE_DICT = {
    'rips': SimplicialVietorisRipsLifting,
    'alpha': SimplicialAlphaComplexLifting
}
LIFT_INV_TYPE_DICT = {
    'rips': InvariantSimplicialVietorisRipsLifting,
    'alpha': SimplicialAlphaComplexLifting 
}

class QM9DataModule(L.LightningDataModule):
    DEBUG_TRAIN = 3
    DEBUG_VAL = 2
    DEBUG_TEST = 1

    def __init__(self, args, batch_size: int = 32):
        super().__init__()
        preproc_str = 'preproc' if args.pre_proc else 'normal'
        self.data_dir = f'./datasets/QM9_delta_{args.dis}_dim_{args.dim}_{args.lift_type}_{preproc_str}'
        self.batch_size = batch_size
        self.debug = args.debug
        self.benchmark = args.benchmark
        TRANSFORM_DICT = LIFT_INV_TYPE_DICT if args.pre_proc else LIFT_TYPE_DICT 
        self.transform = T.Compose([
            InputPreprocTransform(),
            TRANSFORM_DICT[args.lift_type](complex_dim=args.dim, delta=args.dis, feature_lifting='ProjectionElementWiseMean'),
            ])
        self.label_transform = LabelPreprocTransform(target_name=args.target_name)
        self.pre_filter = filter_not_enough_simplices_alpha if args.lift_type == 'alpha' else None
        self.follow = [f"x_{i}" for i in range(args.dim+1)] + ['x']

    def setup(self, stage: str):

        print('Preparing data...')
        if self.benchmark:
            dataset_ = QM9(root=self.data_dir, pre_filter=self.pre_filter)
            dataset = []
            for data in dataset_:
                start_lift_time = time.perf_counter()
                dataset.append(self.transform(data))
                wandb.log({
                    'Lift individual': time.perf_counter() - start_lift_time
                })
                break
        elif self.debug:
            dataset = QM9(root=self.data_dir, pre_filter=self.pre_filter)
            dataset = [self.transform(data) for data in dataset[:self.DEBUG_TRAIN+self.DEBUG_VAL+self.DEBUG_TEST+1]]
        else:
            dataset = QM9(root=self.data_dir, pre_transform=self.transform, pre_filter=self.pre_filter)
            dataset = dataset.shuffle()

        print('Preparing labels...')
        dataset = [self.label_transform(data) for data in tqdm(dataset)]
        print('Preparation done!')

        if self.debug:
            self.qm9_train = dataset[:self.DEBUG_TRAIN]
            self.qm9_val = dataset[self.DEBUG_TRAIN:self.DEBUG_TRAIN+self.DEBUG_VAL]
            self.qm9_test = dataset[self.DEBUG_TRAIN+self.DEBUG_VAL:]
        else:
            n_train, n_test = 100000, 110000
            self.qm9_train = dataset[:n_train]
            self.qm9_test = dataset[n_train:n_test]
            self.qm9_val = dataset[n_test:]

    def train_dataloader(self):
        return DataLoader(self.qm9_train, batch_size=self.batch_size, shuffle=True, follow_batch=self.follow)

    def val_dataloader(self):
        return DataLoader(self.qm9_val, batch_size=self.batch_size, shuffle=False, follow_batch=self.follow)

    def test_dataloader(self):
        return DataLoader(self.qm9_test, batch_size=self.batch_size, shuffle=False, follow_batch=self.follow)