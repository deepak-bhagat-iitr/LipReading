import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio
import gdown
import zipfile

# %% Check for GPU availability
tf.config.list_physical_devices('GPU')
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# %% Download dataset
url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output = 'data.zip'

if not os.path.exists(output):
    print(f"'{output}' not found. Downloading...")
    gdown.download(url, output, quiet=False)
    
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall()
    print("Extraction complete.")
else:
    print(f"'{output}' already exists. Skipping download and extraction.")

# %% Define functions to load data
def load_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame = frame[190:236, 80:220]  # Crop to (46, 140)
        frame = frame[..., np.newaxis]  # Add channel dimension
        frames.append(frame)
    cap.release()
    
    frames = np.array(frames)
    mean = np.mean(frames)
    std = np.std(frames)
    return (frames - mean) / std

# %% Character mappings
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="[UNK]")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="[UNK]", invert=True)

def load_alignments(path: str) -> tf.Tensor:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: tf.Tensor) -> List[tf.Tensor]:
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    
    return frames, alignments

def mappable_function(path: tf.Tensor) -> List[tf.Tensor]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

# %% Create data pipeline
data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75, 46, 140, 1], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

# Reduce the training set size to 50 and the test set size accordingly
train = data.take(50)
test = data.skip(50)

# %% Convert frames to uint8 format for GIF creation
def convert_frames_to_uint8(frames: np.ndarray) -> np.ndarray:
    frames_np = np.array(frames)
    frames_scaled = (frames_np * 255).astype(np.uint8)  # Scale to [0, 255] range and convert to uint8

    if frames_scaled.shape[-1] == 1:
        frames_scaled = np.squeeze(frames_scaled, axis=-1)  # Remove the last channel if it's single-channel (grayscale)

    return frames_scaled

# Sample frame extraction and GIF creation
sample = data.as_numpy_iterator().next()
converted_frames = convert_frames_to_uint8(sample[0][0])  # Convert the frames
imageio.mimsave('./animation.gif', converted_frames, fps=10)  # Save as GIF

# %% Display a sample frame
plt.imshow(sample[0][0][35].squeeze(), cmap='gray')
plt.show()

# %% Neural Network model definition
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

model.summary()

# %% Setup training options and train
def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset) -> None:
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        input_length = np.ones(yhat.shape[0]) * yhat.shape[1]
        decoded = tf.keras.backend.ctc_decode(yhat, input_length)[0][0].numpy()
        
        for x in range(len(yhat)):
            real_text = tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8')
            predicted_text = tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8')
            print('Original:', real_text)
            print('Prediction:', predicted_text)
            print('~'*100)

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

checkpoint_callback = ModelCheckpoint(
    os.path.join('models', 'checkpoint.weights.h5'),  # Correct file extension
    monitor='loss',
    save_weights_only=True
)

schedule_callback = LearningRateScheduler(scheduler)
example_callback = ProduceExample(test)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

# Train the model with the updated callbacks
model.fit(train, validation_data=test, epochs=1, callbacks=[checkpoint_callback, schedule_callback, example_callback])

# %% Make a prediction
model.load_weights('models/checkpoint.weights.h5')  # Ensure the file path is correct

test_data = test.as_numpy_iterator()
sample = test_data.next()
yhat = model.predict(sample[0])

print('~'*100, 'REAL TEXT')
real_texts = [tf.strings.reduce_join(num_to_char(word)).numpy().decode('utf-8') for word in sample[1]]
print(real_texts)

decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75, 75], greedy=True)[0][0].numpy()

print('~'*100, 'PREDICTIONS')
predicted_texts = [tf.strings.reduce_join(num_to_char(word)).numpy().decode('utf-8') for word in decoded]
print(predicted_texts)

# %% Test on a video
sample = load_data(tf.convert_to_tensor('./data/s1/bras9a.mpg'))

print('~'*100, 'REAL TEXT')
real_text = tf.strings.reduce_join([num_to_char(word) for word in sample[1]]).numpy().decode('utf-8')
print(real_text)

yhat = model.predict(tf.expand_dims(sample[0], axis=0))

decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

print('~'*100, 'PREDICTIONS')
predicted_text = tf.strings.reduce_join([num_to_char(word) for word in decoded]).numpy().decode('utf-8')
print(predicted_text)
