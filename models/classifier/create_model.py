import tensorflow as tf
from tensorflow.keras import layers, models

def create_classifier_model():
    """Create a simple CNN model for traffic sign damage classification"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification: damaged or not
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Create and save the model
    model = create_classifier_model()
    model.save('damage_model.h5')
    print("Model created and saved as 'damage_model.h5'") 