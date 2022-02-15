import pytorch_lightning as pl

def get_callbacks(CFG):
    """Returns the callbacks used by pl.Trainer.
    
    Args:
        CFG: python class object as config
        
    Returns:
        callbacks (list): list with callbacks.
    """
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

    if CFG.all_data:
        callbacks = [lr_monitor]
    else:
        callbacks = [lr_monitor, checkpoint_callback, early_stopping_callback]

    return callbacks