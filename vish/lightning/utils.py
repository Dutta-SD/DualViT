from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc_fine",  # Monitor the validation loss
    filename="tpdualvitcifar100-{epoch:02d}-{val_acc_fine:.3f}",  # Checkpoint filename format
    save_top_k=2,  # Save only the best model checkpoint
    mode="max",  # 'min' mode means we want to minimize the monitored metric
)
early_stopping_callback = EarlyStopping(
    monitor="val_acc_fine",  # Monitor the validation loss
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    mode="max",  # 'min' mode means we want to minimize the monitored metric
)
