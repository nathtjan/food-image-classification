import pandas as pd
import time
import torch
from resnet50 import train_resnet_from_df # Ganti ini

# Konfigurasi
IMAGE_BASE_PATH = 'food_cls/'
TRAIN_CSV_PATH = IMAGE_BASE_PATH + 'train.csv'
VAL_CSV_PATH = IMAGE_BASE_PATH + 'val.csv'
SAVE_PATH = 'weights/resnet50_from_scratch.pth'

def fullpath(path):
    return IMAGE_BASE_PATH + path

# Muat data
train_df = pd.read_csv(TRAIN_CSV_PATH)
val_df = pd.read_csv(VAL_CSV_PATH)

train_df["path"] = train_df["path"].apply(fullpath)
val_df["path"] = val_df["path"].apply(fullpath)


training_config = dict(
    batch_size=32,
    epochs=60,      
    lr=0.1,        # <-- Learning rate SGD, bisa dieksperimen
    num_workers=4
)

# Jalankan training
start = time.time()
model, label2dense, dense2label = train_resnet_from_df(
    train_df=train_df,
    val_df=val_df,
    config=training_config
)
end = time.time()

# Simpan model
torch.save(model.state_dict(), SAVE_PATH) 

print(f"Finished training in {end-start:.3f} seconds")