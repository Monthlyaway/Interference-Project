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

    # model = LinearVAE(
    #     seq_len=800,
    #     latent_dim=args.latent_dim,
    #     lr=args.lr,
    #     alpha=args.alpha
    # )

    model = CNNAutoencoder(800, args.latent_dim, args.lr)

    # model = TransformerVAE(
    #     seq_len=800, latent_dim=args.latent_dim, lr=args.lr, alpha=args.alpha)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        filename='cnn_ae_epoch={epoch}-step={step}-val_acc={val/loss:.2f}',
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
        enable_model_summary=True,
        accumulate_grad_batches=2,
    )
    print('Starting training')
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)

    main(parser.parse_args())
