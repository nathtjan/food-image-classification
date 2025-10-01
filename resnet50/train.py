import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .model import create_resnet50_model # Impor dari file model.py
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau


def train_resnet_from_df(train_df, val_df, image_base_path, config):
    # Ekstrak konfigurasi
    img_size = config.get('img_size', (224, 224))
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 20)
    num_classes = train_df['class'].nunique()

    # Buat Data Generator
    datagen = ImageDataGenerator(rescale=1./255.)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_base_path,
        x_col='path', y_col='class',
        target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=image_base_path,
        x_col='path', y_col='class',
        target_size=img_size, batch_size=batch_size, class_mode='categorical'
    )

    # Buat dan compile model
    model = create_resnet50_model(num_classes=num_classes, input_shape=img_size + (3,))
    optimizer_sgd = SGD(learning_rate=0.01, momentum=0.9, decay=0.0001)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6)
    model.compile(optimizer=optimizer_sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # train
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[lr_scheduler]
    )

    return model, history