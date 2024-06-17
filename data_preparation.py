import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def scalar(img):
    return img


def prepare_data(data_dir, batch_size=16, img_size=(224, 224), channels=3):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    df = pd.DataFrame({
        'FILEPATH': filepaths,
        'LABELS': labels
    })

    train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.6, shuffle=True, random_state=123)

    ts_length = len(test_df)
    test_batch_size = max(
        sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))

    tr_gen = ImageDataGenerator(preprocessing_function=scalar)
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)

    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='FILEPATH', y_col='LABELS', target_size=img_size,
                                           class_mode='categorical',
                                           color_mode='rgb', shuffle=True, batch_size=batch_size)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='FILEPATH', y_col='LABELS', target_size=img_size,
                                           class_mode='categorical',
                                           color_mode='rgb', shuffle=True, batch_size=batch_size)

    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='FILEPATH', y_col='LABELS', target_size=img_size,
                                          class_mode='categorical',
                                          color_mode='rgb', shuffle=False, batch_size=test_batch_size)

    return train_gen, valid_gen, test_gen


if __name__ == "__main__":
    DATA_DIR = 'chest_xray/train'
    BATCH_SIZE = 16
    IMG_SIZE = (224, 224)
    CHANNELS = 3

    train_gen, valid_gen, test_gen = prepare_data(DATA_DIR, BATCH_SIZE, IMG_SIZE, CHANNELS)
