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

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

from constant import Constant

class ModelTrainer:
    def preprocess_image(self, img_array, target_size=(224, 224)):
        # Resize the image using NumPy's resize. Note: np.resize and PIL's resize behave differently.
        img_array = np.array(image.smart_resize(img_array, target_size))

        # Normalize the image
        img_array = img_array / 255.0

        # Expand dimensions to fit the model input format
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def load_dataframe(self):
        # Load CSV file
        dataframe = pd.read_csv(Constant.BASE_DIR + '/healthcare/dataset/miner/Data_Entry.csv')

        # Split data into train, validation, and test sets
        train_df, test_df = train_test_split(dataframe, test_size=0.2)
        train_df, val_df = train_test_split(train_df, test_size=0.25)  # 0.25 x 0.8 = 0.2

        train_image_paths = train_df['Image_Index'].values
        train_labels = train_df['Finding_Labels'].values

        val_image_paths = val_df['Image_Index'].values
        val_labels = val_df['Finding_Labels'].values

        train_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_image)
        val_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_image)

        path_to_image_directory = Constant.BASE_DIR + '/healthcare/dataset/miner/images'
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=path_to_image_directory,
            x_col='Image_Index',
            y_col='Finding_Labels',
            class_mode='categorical',  # or 'binary' for binary labels
            target_size=(224, 224),
            batch_size=32
        )

        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=path_to_image_directory,
            x_col='Image_Index',
            y_col='Finding_Labels',
            class_mode='categorical',
            target_size=(224, 224),
            batch_size=32
        )
        num_classes = len(train_generator.class_indices)
        return train_generator, val_generator, train_df, val_df, num_classes

    def load_model(self, num_classes):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            # Add more layers as needed
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')  # num_classes based on your dataset
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        train_generator, val_generator, train_df, val_df, num_classes = self.load_dataframe()
        model = self.load_model(num_classes)
        checkpoint = ModelCheckpoint(
            filepath=Constant.BASE_DIR + '/healthcare/models/model_checkpoint.h5', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            mode='auto'
        )
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_df) // 32,  # Adjust based on your batch size
            epochs=10,  # Number of epochs
            validation_data=val_generator,
            validation_steps=len(val_df) // 32,
            callbacks=[checkpoint]
        )