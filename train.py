import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from data_modules import BaseDataModule
from model_modules import *
import argparse
from lightning.pytorch.cli import LightningCLI
import torch

torch.set_float32_matmul_precision('medium')


def main(args):
    data_module = BaseDataModule(batch_size=args.batch_size)
    data_module.prepare_data()

    # model = LinearVAE(
    #     seq_len=800,
    #     latent_dim=args.latent_dim,
    #     lr=args.lr,
    #     alpha=args.alpha
    # )

    # model = CNNAutoencoder(800, args.latent_dim, args.lr)
    # model = CNNVAE(800, latent_dim=args.latent_dim, lr=args.lr, alpha=args.alpha)
    # model = TransformerAE(800, latent_dim=args.latent_dim, lr=args.lr)
    # model = TransformerVAE(800, latent_dim=args.latent_dim,
    #    lr=args.lr, alpha=args.alpha)
    # model = LinearAE(800, latent_dim=args.latent_dim, lr=args.lr)
    # model = CNNAE(800, latent_dim=args.latent_dim, lr=args.lr)

    # model = TransformerVAE(
    #     seq_len=800, latent_dim=args.latent_dim, lr=args.lr, alpha=args.alpha)
    # model = CNNAEAttention(latent_dim=args.latent_dim, lr=args.lr)
    model = CDMA_Net(lr=args.lr)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        filename='cdma_epoch={epoch}-step={step}-val_loss={val/loss:.2f}',
        auto_insert_metric_name=False,
        save_top_k=2,
        mode='min'
    )

    wandb_logger = WandbLogger(
        project='Satelite-Interference', log_model=True)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        accumulate_grad_batches=2,
        fast_dev_run=False,
    )
    print('Starting training')
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=1.0)

    main(parser.parse_args())
