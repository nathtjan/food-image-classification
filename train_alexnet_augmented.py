import os
import pandas as pd
import time
import torch
from alexnet import train_alexnet_from_df


BASE_PATH = "food_cls"


def fullpath(path):
    return os.path.join(BASE_PATH, path)


train_csv_path = fullpath("augmented.csv")
val_csv_path = fullpath("val.csv")
save_path = "weights/alexnet-augmented.pth"
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# fix path to include base path
train_df["path"] = train_df["path"].apply(fullpath)
val_df["path"] = val_df["path"].apply(fullpath)

# full training configuration
config = dict(
    train_df=train_df,
    val_df=val_df,
    batch_size=32,
    epochs=90,
    lr=0.001,
    num_workers=4
)

start = time.time()
alexnet, label2dense, dense2label = train_alexnet_from_df(**config)
end = time.time()

torch.save(alexnet.state_dict, save_path)

print(f"Finished training in {end-start:.03f} seconds")