import os, random, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


from datasets import Dataset
import torch

from tqdm import tqdm
import nltk
nltk.download('stopwords', quiet=True)

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

print('TF:', tf.__version__)
print('GPU:', len(tf.config.list_physical_devices('GPU')) > 0)


# ##  Configuration — **edit these paths**


# ── EDIT THESE ────────────────────────────────────────────────────────────
CSV_PATH   = 'books_dataset.csv'   # path to CSV
IMAGES_DIR = 'images'              # subfolders = category names
OUTPUT_DIR = 'outputs'                        # saved models & artefacts
# ──────────────────────────────────────────────────────────────────────────

CATEGORIES  = ['baby books', 'cooking', 'japanese', 'kittens']
NUM_CLASSES = len(CATEGORIES)
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
MAX_LEN     = 128    # token length for NLP models
VOCAB_SIZE  = 20_000 # for Keras tokenizer (LSTM / GRU)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print('Config OK — output dir:', os.path.abspath(OUTPUT_DIR))

OUTPUT_DIR = 'outputs'                        # saved models & artefacts
# ──────────────────────────────────────────────────────────────────────────

CATEGORIES  = ['baby books', 'cooking', 'japanese', 'kittens']
NUM_CLASSES = len(CATEGORIES)
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
MAX_LEN     = 128    # token length for NLP models
VOCAB_SIZE  = 20_000 # for Keras tokenizer (LSTM / GRU)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print('Config OK — output dir:', os.path.abspath(OUTPUT_DIR))


augment = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
], name='augmentation')

AUTOTUNE = tf.data.AUTOTUNE

raw_train = image_dataset_from_directory(
    IMAGES_DIR, validation_split=0.2, subset='training',
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)
raw_val = image_dataset_from_directory(
    IMAGES_DIR, validation_split=0.2, subset='validation',
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)

train_img_ds = raw_train.map(
    lambda x, y: (augment(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)
val_img_ds = raw_val.prefetch(AUTOTUNE)

class_names = raw_train.class_names
print('CNN class names:', class_names)


def build_efficientnet_cnn(num_classes=NUM_CLASSES):
    base = EfficientNetB0(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x   = layers.Lambda(eff_preprocess)(inp)
    x   = base(x, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    m   = models.Model(inp, out, name='CNN_EfficientNetB0')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

cnn3 = build_efficientnet_cnn()
cnn3.summary()

def train_model(model, train_ds, val_ds, epochs=15):
    cb = [
        callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor='val_accuracy'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=2, monitor='val_loss'),
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, f'{model.name}_best.keras'),
            save_best_only=True, monitor='val_accuracy'
        )
    ]
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)

print('Training CNN 3 — EfficientNetB0')
h_cnn3 = train_model(cnn3, train_img_ds, val_img_ds)


def fine_tune(model, base_layer_name, train_ds, val_ds, unfreeze=-20, epochs=5):
    base = model.get_layer(base_layer_name)
    base.trainable = True
    for layer in base.layers[:unfreeze]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cb = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, f'{model.name}_finetuned.keras'), save_best_only=True
        )
    ]
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)

# The string must match the base model's internal name
print('Fine-tuning EfficientNetB0')
h_cnn3_ft = fine_tune(cnn3, 'efficientnetb0', train_img_ds, val_img_ds)


def plot_history(h, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(h.history['accuracy'], label='train')
    ax[0].plot(h.history['val_accuracy'], label='val')
    ax[0].set_title(f'{title} Accuracy'); ax[0].legend()
    ax[1].plot(h.history['loss'], label='train')
    ax[1].plot(h.history['val_loss'], label='val')
    ax[1].set_title(f'{title} Loss'); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{title.replace(" ","_")}.png'))
    plt.show()


plot_history(h_cnn3_ft, 'CNN_EfficientNetB0_finetuned.keras')

for name, m in [('resnet50', cnn1), ('MobileNetV2', cnn2), ('EfficientNetB0', cnn3)]:
    loss, acc = m.evaluate(val_img_ds, verbose=0)
    print(f'{name:18s}  val_acc={acc:.4f}  val_loss={loss:.4f}')


for name, model in [('EfficientNetB0', cnn3)]:
    loss, acc = model.evaluate(val_img_ds, verbose=0)
    print(f'{name:20s}  val_acc={acc:.4f}  val_loss={loss:.4f}')

cnn3.save(os.path.join(OUTPUT_DIR, 'cnn3_efficientnet.keras'))

cnn3 = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, 'cnn3_efficientnet.keras'), safe_mode=False)