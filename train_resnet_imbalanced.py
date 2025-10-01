# train_resnet_imbalanced.py
import pandas as pd
import time
from resnet50 import train_resnet_from_df

# Konfigurasi
IMAGE_BASE_PATH = 'food_cls/'

TRAIN_CSV_PATH = 'food_cls/imbalanced.csv' 
VAL_CSV_PATH = 'food_cls/val.csv'

SAVE_PATH = 'weights/resnet50_imbalanced.h5'

# Muat data
train_df = pd.read_csv(TRAIN_CSV_PATH)
val_df = pd.read_csv(VAL_CSV_PATH)

# Konfigurasi training
training_config = dict(
    img_size=(224, 224),
    batch_size=32,
    epochs=60
)

# Jalankan training
start = time.time()
model, history = train_resnet_from_df(
    train_df=train_df,
    val_df=val_df,
    image_base_path=IMAGE_BASE_PATH,
    config=training_config
)
end = time.time()

# Simpan model
model.save(SAVE_PATH)

print(f"Finished training in {end-start:.3f} seconds")
best_val_acc = max(history.history['val_accuracy'])
print(f"Best validation accuracy: {best_val_acc:.4f}")