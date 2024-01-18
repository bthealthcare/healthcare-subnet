# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 demon

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import re
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
import gc
import multiprocessing
from keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import bittensor as bt
from constant import Constant

class CustomModelCheckpoint(Callback):
    def __init__(self, model, path, save_freq, monitor='val_loss'):
        super(CustomModelCheckpoint, self).__init__()
        self.model = model
        self.path = path
        self.save_freq = save_freq
        self.monitor = monitor
        self.best = np.Inf
        self.batch_counter = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        current = logs['loss']
        if self.best == np.Inf:
            self.best = current
        if self.batch_counter % self.save_freq == 0:
            if current is not None and current < self.best:
                bt.logging.info(f"\nBest Model saved!!! {self.best}, {current}")
                self.best = current
                self.model.save(self.path.format(self.batch_counter))

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        user_input_model_type = config.model_type.lower()
        self.model_type = user_input_model_type if user_input_model_type in ['vgg', 'res', 'efficient', 'mobile'] else 'cnn'
        self.device = config.device
        self.training_mode = config.training_mode.lower()

    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        try:
            # Load image
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)

            # Resize the image using NumPy's resize. Note: np.resize and PIL's resize behave differently.
            img_array = np.array(image.smart_resize(img_array, target_size))

            # Normalize the image
            img_array = img_array / 255.0

            # Expand dimensions to fit the model input format
            # img_array = np.expand_dims(img_array, axis=0)

            return img_array
        except Exception as e:
            return "ERROR"

    def generate_data(self, image_paths, labels, batch_size):
        num_samples = len(image_paths)
        while True:
            for offset in range(0, num_samples, batch_size):
                batch_images = []
                batch_labels = labels[offset:offset+batch_size]
                for img_path in image_paths[offset:offset+batch_size]:
                    absolute_path = Constant.BASE_DIR + '/healthcare/dataset/miner/images/' + img_path
                    img = self.load_and_preprocess_image(absolute_path)
                    if isinstance(img, str):
                        continue
                    batch_images.append(img)
                yield np.array(batch_images), np.array(batch_labels)

    # Function to check if an image exists (mock implementation)
    def image_exists(self, image_name, target_size=(224, 224)):
        image_path = Constant.BASE_DIR + '/healthcare/dataset/miner/images/' + image_name
        if not os.path.exists(image_path):
            return False
        return True

    def load_dataframe(self):
        # Load CSV file
        dataframe = pd.read_csv(Constant.BASE_DIR + '/healthcare/dataset/miner/Data_Entry.csv')

        # Preprocess image names and labels of dataframe
        # String list and corresponding image list
        string_list = dataframe['label']
        image_list = dataframe['image_name']

        # Split the strings into individual labels
        split_labels = [set(string.split('|')) for string in string_list]

        # Initialize MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        binary_array_full = mlb.fit_transform(split_labels)

        # Filter out rows where the corresponding image does not exist
        binary_array_filtered = [binary_array_full[i] for i, image in enumerate(image_list) if self.image_exists(image)]
        
        if not binary_array_filtered:
            bt.logging.error("No images found")
            return False, False, False

        binary_array_filtered = np.vstack(binary_array_filtered)
        
        # Filter out rows where the file does not exist
        dataframe['file_exists'] = dataframe['image_name'].apply(lambda x: self.image_exists(x))
        dataframe = dataframe[dataframe['file_exists']]
        dataframe = dataframe.drop(columns=['file_exists'])
        
        image_paths = dataframe['image_name'].values

        train_gen = self.generate_data(image_paths, binary_array_filtered, self.config.batch_size)

        num_classes = binary_array_filtered.shape[1]

        return train_gen, dataframe, num_classes

    def get_model(self, num_classes):
        model_file_path = Constant.BASE_DIR + '/healthcare/models/' + self.model_type
        
        # Check if model exists
        if not self.config.restart:
            if os.path.exists(model_file_path):
                model = load_model(model_file_path)
                bt.logging.info(f"Model loaded")
                return model
            elif self.model_type == 'cnn' and os.path.exists(Constant.BASE_DIR + '/healthcare/models/best_model'):
                model = load_model(Constant.BASE_DIR + '/healthcare/models/best_model')
                bt.logging.info(f"Model loaded")
                return model
        
        if self.model_type == "cnn":
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                MaxPooling2D(2, 2),
                # Add more layers as needed
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='sigmoid')  # num_classes based on your dataset
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        elif self.model_type == "vgg":
            # Load VGG16 pre-trained on ImageNet without the top layer
            base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            if self.training_mode == "fast":
                base_model_vgg.trainable = False  # Freeze the layers

            # Add custom layers
            x = GlobalAveragePooling2D()(base_model_vgg.output)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(num_classes, activation='softmax')(x)

            model = Model(inputs=base_model_vgg.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        elif self.model_type == "res":
            base_model_res = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            if self.training_mode == "fast":
                base_model_res.trainable = False  # Freeze the layers

            # Add custom layers
            x = GlobalAveragePooling2D()(base_model_res.output)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(num_classes, activation='softmax')(x)

            model = Model(inputs=base_model_res.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        elif self.model_type == "efficient":
            base_model_efficient = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            if self.training_mode == "fast":
                base_model_efficient.trainable = False  # Freeze the layers

            # Add custom layers
            x = GlobalAveragePooling2D()(base_model_efficient.output)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(num_classes, activation='softmax')(x)

            model = Model(inputs=base_model_efficient.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        elif self.model_type == "mobile":
            base_model_mobile = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            if self.training_mode == "fast":
                base_model_mobile.trainable = False  # Freeze the layers

            # Add custom layers
            x = GlobalAveragePooling2D()(base_model_mobile.output)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(num_classes, activation='softmax')(x)

            model = Model(inputs=base_model_mobile.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        # Set device
        if self.device.startswith('cpu'):
            # Do not allow gpus
            tf.config.set_visible_devices([], 'GPU')

            # Set the number of cpu cores
            num_cpu_cores = multiprocessing.cpu_count()
            
            try:
                # Find device numbers from string
                numbers_part = self.device.split(":")
                num_cores_to_use = min(num_cpu_cores, int(numbers_part[1]))
            except Exception as e:
                num_cores_to_use = num_cpu_cores

            # Set TensorFlow's parallelism threads
            tf.config.threading.set_intra_op_parallelism_threads(num_cores_to_use)
            tf.config.threading.set_inter_op_parallelism_threads(num_cores_to_use)

        elif self.device.startswith('gpu'):
            # Find all avaiable gpus
            gpus = tf.config.experimental.list_physical_devices('GPU')

            try:
                # Find device numbers from string
                numbers_part = self.device.split(":")
                numbers = re.findall(r'\d+', numbers_part[1])
                device_numbers = [int(num) for num in numbers if int(num) < len(gpus)]
            except Exception as e:
                device_numbers = []

            if not device_numbers:
                device_numbers = [i for i in range(len(gpus))]

            # Set gpus to use
            if gpus:
                try:
                    selected_gpus = [gpus[i] for i in device_numbers if i < len(gpus)]

                    if selected_gpus:
                        tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')

                        for gpu in selected_gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    selected_gpus = []

        train_generator, train_df, num_classes = self.load_dataframe()

        if train_generator == False:
            return
            
        model = self.get_model(num_classes)

        checkpoint = ModelCheckpoint(
            filepath=Constant.BASE_DIR + '/healthcare/models/' + self.model_type, 
            monitor='loss', 
            verbose=1, 
            save_best_only=True, 
            mode='auto'
        )

        custom_checkpoint = CustomModelCheckpoint(
            model,
            path=Constant.BASE_DIR + '/healthcare/models/' + self.model_type,
            save_freq=self.config.save_model_period  # Change this to your preferred frequency
        )

        # Add EarlyStopping
        early_stopping = EarlyStopping(monitor='loss', patience=10)

        if self.config.num_epochs == -1:
            while True:
                history = model.fit(
                    train_generator,
                    steps_per_epoch=len(train_df) // self.config.batch_size,  # Adjust based on your batch size
                    epochs=1,  # Number of epochs
                    callbacks=[checkpoint, early_stopping]
                )
                K.clear_session()
                gc.collect()
        else:
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_df) // self.config.batch_size,  # Adjust based on your batch size
                epochs=self.config.num_epochs,  # Number of epochs
                callbacks=[checkpoint, early_stopping]
            )