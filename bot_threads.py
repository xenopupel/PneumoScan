import telebot
import numpy as np
from keras.models import load_model
from PIL import Image
from io import BytesIO
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from concurrent.futures import ThreadPoolExecutor

API_TOKEN = ''
bot = telebot.TeleBot(API_TOKEN)

# Загрузка модели
model1 = load_model('eff_model.h5')


def get_image_generator():
    # Эта функция создает и возвращает ImageDataGenerator
    return ImageDataGenerator()


def prepare_image_with_generator(image_bytes):
    # Сохраняем изображение временно
    image = Image.open(BytesIO(image_bytes))
    temp_path = 'temp_image.jpg'  # Используем текущую директорию для простоты
    image.save(temp_path)

    # Создаем временный DataFrame с путем к изображению
    df = pd.DataFrame({'filepaths': [temp_path]})

    # Используем ImageDataGenerator для создания генератора данных
    generator = get_image_generator().flow_from_dataframe(
        dataframe=df,
        x_col='filepaths',
        y_col=None,
        target_size=(224, 224),
        class_mode=None,
        batch_size=1,
        shuffle=False
    )
    return next(generator)  # Используем функцию next() для получения данных


def process_image(image_bytes):
    try:
        # Подготовка изображения
        img = prepare_image_with_generator(image_bytes)

        # Предсказание модели
        prediction1 = model1.predict(img)[0]
        print(prediction1)
        # Вывод результатов
        probability1 = prediction1[1] * 100  # Предполагается, что класс 1 соответствует наличию пневмонии
        return f'Вероятность пневмонии: {probability1:.2f}%'
    except Exception as e:
        return f'Ошибка при обработке изображения: {str(e)}'

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_image, downloaded_file)
            response = future.result()

        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f'Ошибка при обработке изображения: {str(e)}')


if __name__ == '__main__':
    bot.polling()
