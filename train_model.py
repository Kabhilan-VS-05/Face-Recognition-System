import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model():
    data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = data_gen.flow_from_directory('faces', target_size=(100, 100), batch_size=32, color_mode='grayscale', subset='training')
    val_data = data_gen.flow_from_directory('faces', target_size=(100, 100), batch_size=32, color_mode='grayscale', subset='validation')

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(train_data.num_classes, activation='softmax')  # One output per user
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save('face_recognition_model.h5')
    print("Model trained and saved as face_recognition_model.h5")

if __name__ == "__main__":
    train_model()
