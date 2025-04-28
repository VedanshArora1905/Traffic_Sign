import tensorflow as tf
from tensorflow.keras import layers, models

def create_segmenter_model():
    """Create a simple U-Net model for traffic sign damage segmentation"""
    # Input layer
    inputs = layers.Input(shape=(256, 256, 3))
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Decoder
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    
    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Create and save the model
    model = create_segmenter_model()
    model.save('unet_model.h5')
    print("Model created and saved as 'unet_model.h5'") 