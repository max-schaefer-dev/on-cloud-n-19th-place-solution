import pytorch_lightning as pl

def get_callbacks(CFG):
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="jaccardIndex", mode="max", verbose=True, save_top_k=3
    )

    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="jaccardIndex",
        patience=(CFG.patience * 3),
        mode="max",
        verbose=True,
    )

    callbacks = [lr_monitor, checkpoint_callback, early_stopping_callback]
    
    return callbacks