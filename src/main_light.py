import argparse
import os
import torch
import wandb
import copy
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
import time


from modules.models.simplicial.empsn import EMPSN
from src.light_empsn import LitEMPSN
from src.lit_datamodule import QM9DataModule

from src.data_utils import generate_loaders_qm9, calc_mean_mad
from src.utils import set_seed

num_input = 15
num_out = 1

inv_dims = {
    'rank_0': {
        'rank_0': 3,
        'rank_1': 3,
    },
    'rank_1': {
        'rank_1': 6,
        'rank_2': 6,
    }
}

def main(args):
    # # Generate model
    model = EMPSN(
            in_channels=num_input,
            hidden_channels=args.num_hidden,
            out_channels=num_out,
            n_layers=args.num_layers,
            max_dim=args.dim,
            inv_dims=inv_dims
        ).to(args.device)

    # Setup wandb
    wandb.init(project=f"QM9-{args.target_name}-{args.lift_type}-{'preproc' if args.pre_proc else 'no-preproc'}")
    wandb.config.update(vars(args))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')

    seed_everything(args.seed)
    # # Get loaders
    start_lift_time = time.perf_counter()
    qm9_datamodule = QM9DataModule(args, batch_size=args.batch_size)
    # TODO: FIX could be better way to have calc_mean_mad inside LitEMPSN
    #qm9_datamodule.prepare_data()
    qm9_datamodule.setup(stage='fit')

    end_lift_time = time.perf_counter()
    wandb.log({
        'Lift time': end_lift_time - start_lift_time
    })

    mean, mad = calc_mean_mad(qm9_datamodule.train_dataloader())
    mean, mad = mean.to(args.device), mad.to(args.device)

    print('Almost at training...')

    wandb_logger = WandbLogger()
    ckpt_folder = 'models/'

    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    models = os.listdir(ckpt_folder)
    best_model = None
    if len(models):
        best_epoch = -1
        for _model in models:
            if 'latest' in _model:
                curr_epoch = int(_model.split('-')[1].split('=')[1])
                if curr_epoch > best_epoch:
                    best_epoch = curr_epoch
                    best_model = _model
        if not best_model:
            best_model = models[0]
        best_model = os.path.join(ckpt_folder, best_model) 

    best_checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, dirpath='models', filename='empsn-{epoch}-{val_loss:.2f}')
    latest_checkpoint = L.pytorch.callbacks.ModelCheckpoint(monitor='epoch', mode='max', save_top_k=1, dirpath='models', filename='latest-{epoch}-{step}', every_n_epochs=20)

    empsn = LitEMPSN(model, train_samples=len(qm9_datamodule.train_dataloader().dataset),
                      validation_samples=len(qm9_datamodule.val_dataloader().dataset),
                      test_samples=len(qm9_datamodule.test_dataloader().dataset),
                       mae=mad, mad=mad, mean=mean, lr=args.lr, weight_decay=args.weight_decay)
    # state_dict = torch.load(best_model)
    # empsn.load_state_dict(state_dict['state_dict'])
    trainer = L.Trainer(callbacks=[best_checkpoint, latest_checkpoint],deterministic=True, max_epochs=args.epochs,
                        gradient_clip_val=args.gradient_clip, enable_checkpointing=True,
                        accelerator=args.device, devices=1, logger=wandb_logger)# accelerator='gpu', devices=1)

    #trainer.tune(model)


    #tuner = L.pytorch.tuner.Tuner(trainer)
    #tuner.scale_batch_size(empsn, mode='binsearch', datamodule=qm9_datamodule)

    trainer.fit(empsn, datamodule=qm9_datamodule, ckpt_path=best_model)
    trainer.test(empsn, datamodule=qm9_datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode')
    parser.add_argument('--benchmark', action='store_true',
                        help='benchmark mode')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='empsn',
                        help='model')
    parser.add_argument('--num_hidden', type=int, default=77,
                        help='hidden features')
    parser.add_argument('--num_layers', type=int, default=7,
                        help='number of layers')
    parser.add_argument('--act_fn', type=str, default='silu',
                        help='activation function')
    parser.add_argument('--lift_type', type=str, default='rips',
                        help='lift type')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16,
                        help='learning rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='gradient clipping')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='dataset')
    parser.add_argument('--target_name', type=str, default='H',
                        help='regression task')
    parser.add_argument('--dim', type=int, default=2,
                        help='ASC dimension')
    parser.add_argument('--dis', type=float, default=3.0,
                        help='radius Rips complex')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--pre_proc', action='store_true',
                        help='preprocessing')



    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(parsed_args.seed)
    main(parsed_args)