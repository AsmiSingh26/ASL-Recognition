import tensorflow as tf

def build_dual_input_model(num_classes=29):
    # Branch A: Image CNN
    image_input = tf.keras.Input(shape=(224, 224, 3), name="image_input")
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    x = base_model(image_input, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Branch B: Landmark MLP
    landmark_input = tf.keras.Input(shape=(63,), name="landmark_input")
    y = tf.keras.layers.Dense(128, activation="relu")(landmark_input)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dense(64, activation="relu")(y)
    y = tf.keras.layers.Dropout(0.3)(y)

    # Fusion
    combined = tf.keras.layers.Concatenate()([x, y])
    combined = tf.keras.layers.Dense(128, activation="relu")(combined)
    combined = tf.keras.layers.Dropout(0.2)(combined)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(combined)

    model = tf.keras.Model(
        inputs=[image_input, landmark_input],
        outputs=output
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model