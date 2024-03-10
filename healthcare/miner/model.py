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
import sys
import time
import asyncio
from contextlib import contextmanager
import numpy as np
import tensorflow as tf
import multiprocessing
from keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, MobileNet
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from huggingface_hub import HfApi, upload_file, HfFolder, update_repo_visibility

import bittensor as bt
from healthcare.dataset.dataset import load_dataset, load_and_preprocess_image
from healthcare.utils.chain import Chain
from constants import BASE_DIR, ALL_LABELS
from dotenv import load_dotenv
load_dotenv()


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class UploadModelCallback(Callback):
    def __init__(self, monitor, model_directory, access_token, chain):
        self.monitor = monitor
        self.best = float('inf')
        self.model_directory = model_directory
        self.access_token = access_token
        self.chain = chain
        HfFolder.save_token(access_token)
        try:
            load_dotenv()
            api = HfApi()
            username = api.whoami(access_token)["name"]
            self.repo_id = username + "/" + os.getenv('REPO_ID')
            api.create_repo(token=access_token, repo_id=self.repo_id, exist_ok = True)
        except Exception as e:
            bt.logging.error(f"❌ Error occured while creating a repository : {e}")

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current < self.best:
            self.best = current
            # Save the best model
            self.model.save(self.model_directory)

            # Upload the model to hugging face
            HfFolder.save_token(self.access_token)
            try:
                # Make the repository as private before uploading
                update_repo_visibility(self.repo_id, private = True)

                # Upload the model to hugging face
                for root, dirs, files in os.walk(self.model_directory):
                    for file in files:
                        # Generate the full path and then remove the base directory part
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, self.model_directory)
                        with suppress_stdout_stderr():
                            upload_file(
                                path_or_fileobj=full_path,
                                path_in_repo=relative_path,
                                repo_id=self.repo_id
                            )
                bt.logging.info(f"✅ Best model uploaded at {self.repo_id}")

                # Retrieve the latest commit information
                api = HfApi()
                repo_info = api.repo_info(repo_id=self.repo_id, token=self.access_token)
                last_commit_hash = repo_info.sha
                
                # Push the metadata to the chain
                data = " ".join([self.repo_id, last_commit_hash])
                while True:
                    try:
                        asyncio.run(self.chain.store_metadata(data))
                        bt.logging.info("✅ Stored the model to the chain.")
                        break
                    except Exception as e:
                        bt.logging.info(f"❌ Error occured while pushing the model to chain : {e}")
                        time.sleep(12)
                        continue

            except Exception as e:
                print(f"❌ Error occured while pushing the model : {e}")
            
            # Make the repository as public
            update_repo_visibility(self.repo_id, private = False)

class ModelTrainer:
    def __init__(self, neuron):
        self.neuron = neuron
        self.config = neuron.config
        user_input_model_type = self.config.model_type.lower()
        self.model_type = user_input_model_type if user_input_model_type in ['vgg', 'res', 'efficient', 'mobile', 'vit'] else 'cnn'
        self.device = self.config.device
        self.training_mode = self.config.training_mode.lower()

    def generate_data(self, image_paths, labels, batch_size):
        num_samples = len(image_paths)
        while True:
            for offset in range(0, num_samples, batch_size):
                batch_images = []
                batch_labels = labels[offset:offset+batch_size]
                current_batch_size = 0
                for img_path in image_paths[offset:offset+batch_size]:
                    img = load_and_preprocess_image(os.path.join(BASE_DIR, 'healthcare/dataset/miner/images', img_path))
                    if isinstance(img, str):
                        continue
                    batch_images.append(img)
                    current_batch_size += 1
                if current_batch_size == batch_size:
                    yield np.array(batch_images), np.array(batch_labels)

    def load_dataframe(self):
        csv_path = os.path.join(BASE_DIR, 'healthcare/dataset/miner/Data_Entry.csv')
        image_dir = os.path.join(BASE_DIR, 'healthcare/dataset/miner/images')
        image_list, binary_output, dataframe = load_dataset(csv_path, image_dir)
                
        if not binary_output:
            bt.logging.error("❌ No images found.")
            return False, False, False

        train_gen = self.generate_data(image_list, binary_output, self.config.batch_size)

        all_labels_list = set(ALL_LABELS.split('|'))
        num_classes = len(all_labels_list)

        return train_gen, dataframe, num_classes

    def get_model(self, num_classes):
        model_file_path = os.path.join(BASE_DIR, 'healthcare/models', self.model_type)
        
        # Check if model exists
        if not self.config.restart and os.path.exists(model_file_path):
            model = load_model(model_file_path)
            bt.logging.info(f"🔃 Model loaded")
            return model
        
        if self.model_type == "cnn":
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='sigmoid')
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
            predictions = Dense(num_classes, activation='sigmoid')(x)

            model = Model(inputs=base_model_vgg.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        elif self.model_type == "res":
            base_model_res = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            if self.training_mode == "fast":
                base_model_res.trainable = False  # Freeze the layers

            # Add custom layers
            x = GlobalAveragePooling2D()(base_model_res.output)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(num_classes, activation='sigmoid')(x)

            model = Model(inputs=base_model_res.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        elif self.model_type == "efficient":
            base_model_efficient = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            if self.training_mode == "fast":
                base_model_efficient.trainable = False  # Freeze the layers

            # Add custom layers
            x = GlobalAveragePooling2D()(base_model_efficient.output)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(num_classes, activation='sigmoid')(x)

            model = Model(inputs=base_model_efficient.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        elif self.model_type == "mobile":
            base_model_mobile = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            if self.training_mode == "fast":
                base_model_mobile.trainable = False  # Freeze the layers

            # Add custom layers
            x = GlobalAveragePooling2D()(base_model_mobile.output)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(num_classes, activation='sigmoid')(x)

            model = Model(inputs=base_model_mobile.input, outputs=predictions)
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):        
        # Define EarlyStopping
        early_stopping = EarlyStopping(monitor='loss', patience=10)

        # Define upload_callback
        access_token = os.getenv('ACCESS_TOKEN')
        if not access_token:
            bt.logging.error(f"❌ Define ACCESS_TOKEN in .env file.")
            return

        model_directory = os.path.join(BASE_DIR, 'healthcare/models', self.model_type)
        chain = Chain(self.config.netuid, self.neuron.subtensor, self.neuron.wallet)

        upload_callback = UploadModelCallback(
            monitor='loss',
            model_directory=model_directory,
            access_token=access_token,
            chain = chain
        )

        # Load required details from dataframe and define model architecture
        train_generator, train_df, num_classes = self.load_dataframe()
        if train_generator == False:
            return
        model = self.get_model(num_classes)

        # Start training model
        if self.config.num_epochs == -1:
            while True:
                history = model.fit(
                    train_generator,
                    steps_per_epoch=len(train_df) // self.config.batch_size,  # Adjust based on your batch size
                    epochs=1,  # Number of epochs
                    callbacks=[early_stopping, upload_callback],
                    verbose=2
                )
        else:
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_df) // self.config.batch_size,  # Adjust based on your batch size
                epochs=self.config.num_epochs,  # Number of epochs
                callbacks=[early_stopping, upload_callback],
                verbose=2
            )