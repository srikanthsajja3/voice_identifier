from tensorflow import keras
from tensorflow.keras import layers, models

def create_model(input_shape=(128, 173, 1), num_classes=10):
    """
    Enhanced CNN model for maximum accuracy on UrbanSound8K.
    Includes an extra Conv block while remaining compatible with XAI.
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 3 (Target for XAI)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Fully Connected Block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output_layer")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use a slightly lower learning rate for better stability on deep models
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_trained_model(path):
    return models.load_model(path)
