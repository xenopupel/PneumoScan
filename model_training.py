import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop, Adamax
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers


def build_base_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3,3), strides=1, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=2, padding='same'))
    model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=2, padding='same'))
    model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=2, padding='same'))
    model.add(Conv2D(128, (3,3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=2, padding='same'))
    model.add(Conv2D(256, (3,3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_efficientnet_model(input_shape, num_classes):
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape, pooling='max')
    base_model.trainable = False
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_resnet_model(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape, pooling='max')
    base_model.trainable = False
    model = Sequential([
        base_model,
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_densenet_model(input_shape, num_classes):
    base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape, pooling='max')
    base_model.trainable = False
    model = Sequential([
        base_model,
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_and_save_model(model, train_gen, valid_gen, epochs, model_path):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_gen, epochs=epochs, validation_data=valid_gen, callbacks=[learning_rate_reduction])
    model.save(model_path)


if __name__ == "__main__":
    from data_preparation import prepare_data

    DATA_DIR = 'chest_xray/train'
    BATCH_SIZE = 16
    IMG_SIZE = (224, 224)
    CHANNELS = 3
    EPOCHS = 12

    train_gen, valid_gen, test_gen = prepare_data(DATA_DIR, BATCH_SIZE, IMG_SIZE, CHANNELS)
    num_classes = len(train_gen.class_indices)
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)

    # Base CNN Model
    base_cnn_model = build_base_cnn_model(input_shape, num_classes)
    train_and_save_model(base_cnn_model, train_gen, valid_gen, EPOCHS, 'base_cnn.h5')

    # EfficientNet Model
    efficientnet_model = build_efficientnet_model(input_shape, num_classes)
    train_and_save_model(efficientnet_model, train_gen, valid_gen, EPOCHS, 'eff_model.h5')

    # ResNet Model
    resnet_model = build_resnet_model(input_shape, num_classes)
    train_and_save_model(resnet_model, train_gen, valid_gen, EPOCHS, 'resnet_model.h5')

    # DenseNet Model
    densenet_model = build_densenet_model(input_shape, num_classes)
    train_and_save_model(densenet_model, train_gen, valid_gen, EPOCHS, 'densenet_model.h5')
