
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


# CNN — image data pipeline with augmentation


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


from tensorflow.keras.applications import ResNet50

def build_resnet_cnn(num_classes=NUM_CLASSES):
    base = ResNet50(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False

    inp = layers.Input(shape=(*IMG_SIZE, 3))
    # ResNet50 expects [-1, 1] range same as MobileNetV2
    x   = layers.Rescaling(scale=1./127.5, offset=-1)(inp)
    x   = base(x, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    m   = models.Model(inp, out, name='CNN_ResNet50')
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return m

cnn1 = build_resnet_cnn()
cnn1.summary()

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

print('Training CNN 1 — ResNet50')
h_cnn1 = train_model(cnn1, train_img_ds, val_img_ds)

base = cnn1.get_layer('resnet50')
base.trainable = True

# Freeze everything except the last 50 layers
for layer in base.layers[:-50]:
    layer.trainable = False

print(f"Trainable layers: {sum(1 for l in base.layers if l.trainable)}/{len(base.layers)}")

cnn1.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # much lower LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

h_cnn1_ft = cnn1.fit(
    train_img_ds,
    validation_data=val_img_ds,
    epochs=20,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        callbacks.ReduceLROnPlateau(factor=0.3, patience=3, monitor='val_loss'),
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'cnn1_resnet50_best.keras'),
            save_best_only=True, monitor='val_accuracy'
        )
    ]
)

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

plot_history(h_cnn1_ft, 'cnn1_resnet50_best.keras')


for name, m in [('resnet50', cnn1)]:
    loss, acc = m.evaluate(val_img_ds, verbose=0)
    print(f'{name:18s}  val_acc={acc:.4f}  val_loss={loss:.4f}')


for name, model in [('ResNet50', cnn1)]:
    loss, acc = model.evaluate(val_img_ds, verbose=0)
    print(f'{name:20s}  val_acc={acc:.4f}  val_loss={loss:.4f}')

cnn1.save(os.path.join(OUTPUT_DIR, 'cnn1_resnet50.keras'))

cnn1 = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, 'cnn1_resnet50.keras'), safe_mode=False)