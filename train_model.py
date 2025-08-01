from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.preprocess import preprocess
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_encoder = preprocess(limit_per_class=500)

    # Reshape images to (64, 64, 1)
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)

    # One-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Classes: {label_encoder.classes_}")

    # Define model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')
    ])

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)


    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(datagen.flow(X_train, y_train, batch_size=64),
          epochs=30,
          validation_data=(X_test, y_test),
          callbacks=[early_stop, model_checkpoint])


    model.save("asl_model.h5")
    print(" Model saved to asl_model.h5")
