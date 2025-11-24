import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 64
EPOCHS = 50

def build_model():
    """
    Defines the CNN Architecture for Emotion Detection (FER-2013).
    Layers: 4 Convolutional Blocks + 2 Dense Layers
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Dense Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax')) # 7 Emotions

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # This script is used for training the model on the dataset
    print("Initializing Model Architecture...")
    model = build_model()
    model.summary()
    print("Model ready for training.")
    # model.fit(...) # Training logic would go here
    # model.save('model_weights.h5')
