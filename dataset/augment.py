import albumentations as A

def prepare_train_augmentation():

    tranform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

    return tranform

def prepare_val_augmentation():
    """

    """

    tranform = A.Compose([
    ])

    return None