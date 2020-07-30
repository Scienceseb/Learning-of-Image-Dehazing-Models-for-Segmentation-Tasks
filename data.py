from os.path import join

from dataset_seg3 import DatasetFromFolder_3


def get_training_set(root_dir):
    train_dir = join(root_dir, "train")

    return DatasetFromFolder_3(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")

    return DatasetFromFolder_3(test_dir)

def get_val_set(root_dir):
    val_dir = join(root_dir, "val")

    return DatasetFromFolder_3(val_dir)