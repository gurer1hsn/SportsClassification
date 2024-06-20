# SportsClassification
!pip install tensorflow gitpython
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = '/content/drive/MyDrive/SportsDataset'


batch_size = 32
img_height = 180
img_width = 180

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    data_dir,
    subset="training",
    seed=123,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_data = train_datagen.flow_from_directory(
    data_dir,
    subset="validation",
    seed=123,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

!git config --global user.email "hasanhuseyingurer06@gmail.com"
!git config --global user.name "Hasan Hüseyin"

# Adım 1: Google Drive'ı Bağlama
from google.colab import drive
drive.mount('/content/drive')

# Adım 2: Veri Setini Yükleme ve İşleme
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_dir = '/content/drive/My Drive/SportsDataset'

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# steps_per_epoch ve validation_steps hesaplanması
steps_per_epoch = np.ceil(train_generator.samples / train_generator.batch_size)
validation_steps = np.ceil(validation_generator.samples / validation_generator.batch_size)

# Adım 3: CNN Modeli Oluşturma ve Eğitme
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=15)

# Modeli kaydetme
model.save('/content/drive/My Drive/SportsDataset/sports_classification_model.h5')

# Adım 4: Modeli GitHub'a Yükleme
# GitHub depoyu klonlama
!git clone https://github.com//hasanhuseyingurer/SportsClassification.git
%cd depo_adi

# Git ayarları
!git config --global user.email "hasanhuseyingurer06@gmail.com"
!git config --global user.name "Hasan Hüseyin"

# Dosyaları kopyalama
!cp /content/drive/My\ Drive/SportsDataset/sports_classification_model.h5 .

# GitHub'a yükleme
!git add .
!git commit -m "Initial commit"
# Token'ınızı kullanarak push komutunu çalıştırın
token = "WEST"
!git remote set-url origin https://{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}@github.com/hasanhuseyingurer/SportsClassification.git
!git push origin master

!git clone https://github.com/hasanhuseyingurer/SportsClassification.git
%cd SportsClassification

# Mevcut klasörü silme
!rm -rf /content/SportsClassification

# GitHub deposunu yeniden klonlama
!git clone https://github.com/hasanhuseyingurer/SportsClassification.git
%cd SportsClassification

!git clone https://github.com/hasanhuseyingurer/SportsClassification.git
%cd SportsClassification

# Mevcut klasörü silme
!rm -rf /content/SportsClassification

# GitHub deposunu yeniden klonlama
!git clone https://github.com/hasanhuseyingurer/SportsClassification.git
%cd SportsClassification

import os
os.chdir('/content')  # Ana dizine dönün

# Git ayarları
!git config --global user.email "hasanhuseyingurer06@gmail.com"
!git config --global user.name "Hasan Hüseyin"

# Dosyaları ekleme ve GitHub'a gönderme
!git add .
!git commit -m "Modeli ve eğitim çıktılarını ekledim"
!git push origin master

# Mevcut dalı kontrol etme
!git branch

# main dalına geçme (eğer farklı bir daldaysanız)
!git checkout main

# Dosyaları ekleme ve commit etme
!git add .
!git commit -m "Modeli ve eğitim çıktılarını ekledim"

# Değişiklikleri main dalına gönderme
!git push origin main


# Dosyaları ekleme ve commit etme (eğer varsa)
!git add .
!git commit -m "Modeli ve eğitim çıktılarını ekledim"

# Değişiklikleri main dalına gönderme
!git push origin main

from getpass import getpass
import os
from git import Repo

# Get the GitHub token
token = getpass('GitHub Token: ')

# Define the repository URL and directory
repo_url = 'https://<username>:' + token + '@github.com/hasanhuseyingurer/SportsClassification.git'
repo_dir = '/content/repo'

# Clone the repository if it does not already exist
if not os.path.exists(repo_dir):
    Repo.clone_from(repo_url, repo_dir)

# Save your model (assuming 'model' is already defined and trained)
model.save(os.path.join(repo_dir, 'sports_classifier_model.h5'))

