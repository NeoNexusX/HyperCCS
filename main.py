import os
import time
import torch
import pytorch_lightning as pl
from model.layers.main_layer import LightningModule
from model.args import parse_args
from data_prepare.tokenizer import MolTranBertTokenizer
from data_prepare.data_loader import PropertyPredictionDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import seed
from pytorch_lightning.callbacks import EarlyStopping

def main():
    margs = parse_args()
    print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")

    run_name_fields = [
        margs.project_name,
        margs.dataset_name,
        "lr",
        margs.lr_start,
        "batch",
        margs.batch_size,
        "drop",
        margs.dropout,
    ]

    run_name = "_".join(map(str, run_name_fields))

    datamodule = PropertyPredictionDataModule(margs)
    margs.dataset_names = "valid test".split()

    checkpoints_folder = margs.checkpoints_folder
    checkpoint_root = checkpoints_folder
    margs.checkpoint_root = checkpoint_root
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, "models")
    results_dir = os.path.join(checkpoint_root, "results")
    margs.results_dir = results_dir
    margs.checkpoint_dir = checkpoint_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        period=1, 
        save_last=True, 
        dirpath=checkpoint_dir, 
        filename='checkpoint', 
        verbose=True
    )
    
    # 定义 ModelCheckpoint 回调
    r2_checkpoint_callback =  pl.callbacks.ModelCheckpoint(
        monitor="CCS_valid_r2",
        mode="max",
        save_top_k=3,
        filename="best-model-{epoch:02d}-{CCS_valid_r2:.4f}"
    )

    # early stop
    early_stopping_callback = EarlyStopping(
        monitor="CCS_valid_loss",
        patience=10,  # 10个epoch内没有改进就停止
        verbose=True,
        mode="min"    # 因为是监控loss，所以模式为min
    )

    logger = TensorBoardLogger(
        save_dir=checkpoint_root,
        version=run_name,
        name="lightning_logs",
        default_hp_metric=True,
    )
     
    # tokenlizer 设置
    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    seed.seed_everything(margs.seed)

    last_checkpoint_file = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.isfile(last_checkpoint_file):
        print(f"Resuming training from: {last_checkpoint_file}")
        model = LightningModule.load_from_checkpoint(
            last_checkpoint_file,
            config=margs,
            tokenizer=tokenizer
        )
        resume_from_checkpoint = last_checkpoint_file
    elif margs.seed_path:
        print(f"Loaded pre-trained model from {margs.seed_path}")
        model = LightningModule.load_from_checkpoint(
            margs.seed_path,
            strict=False,
            config=margs,
            tokenizer=tokenizer,
            vocab=len(tokenizer.vocab)
        )
        resume_from_checkpoint = None  # 不恢复优化器状态
    else:
        print("Training from scratch")
        model = LightningModule(margs, tokenizer)
        resume_from_checkpoint = None

    # 模型训练器
    trainer = pl.Trainer(
        max_epochs=margs.max_epochs,
        default_root_dir=checkpoint_root,
        gpus=1,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_callback=checkpoint_callback,
        callbacks = [r2_checkpoint_callback, early_stopping_callback],
        num_sanity_val_steps=0,
    )
 
    tic = time.perf_counter()
    trainer.fit(model, datamodule)
    toc = time.perf_counter()
    print('Time was {}'.format(toc - tic))


if __name__ == '__main__':
    main()