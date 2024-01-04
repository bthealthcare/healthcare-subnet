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

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
import numpy as np
import pandas as pd
import os
import base64
from io import BytesIO

import bittensor as bt
import healthcare
from constant import Constant

# Preprocess an image
def preprocess_image(pil_img, target_size=(224, 224)):
    # Resize the image
    resized_img = pil_img.resize(target_size)

    # Convert to RGB if not already in RGB mode
    if resized_img.mode != 'RGB':
        resized_img = resized_img.convert('RGB')

    # Convert the PIL image to a numpy array
    img_array = np.array(resized_img)

    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = img_array / 255.0

    return img_array

def i2t(self, synapse: healthcare.protocol.Predict) -> str:
    """
    It is responsible for diagnosing disease with given image.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # Get the file path of model
    model_file_path = Constant.BASE_DIR + "/healthcare/models/best_model"

    # Check if model exists
    if not os.path.exists(model_file_path):
        return ""

    # Load and preprocess image of synapse
    img_byte = base64.b64decode(synapse.input_image)
    pil_img = Image.open(BytesIO(img_byte))
    processed_image = preprocess_image(pil_img)
    
    # Load the model
    model = load_model(model_file_path)

    # Load labels from Data_Entry
    df = pd.read_csv(Constant.BASE_DIR + '/healthcare/dataset/miner/Data_Entry.csv')
    split_labels = df['label'].apply(lambda x: x.split('|'))

    # Flatten the list of lists to get a single list containing all labels
    all_labels = [label for sublist in split_labels for label in sublist]

    # Get the unique labels
    unique_labels = set(all_labels)

    # If you need it as a list
    unique_labels_sorted_list = sorted(list(unique_labels))

    # Get the predicted label name
    predicted = model.predict(processed_image)
    bt.logging.info(f"{predicted}")
    scores = list(predicted[0])

    # Pair each score with its corresponding label
    score_label_pairs = zip(scores, unique_labels_sorted_list)

    # Filter out labels with scores above the threshold
    labels_above_threshold = [label for score, label in score_label_pairs if score > self.config.threshold]

    # If no labels are above the threshold, find the label with the highest score
    if not labels_above_threshold:
        # Get the index of the highest score
        max_score_index = scores.index(max(scores))
        # Return the label with the highest score
        labels_above_threshold.append(unique_labels_sorted_list[max_score_index])

    return '|'.join(labels_above_threshold)