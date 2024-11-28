import requests
import os

BASE_URL = "http://127.0.0.1:5000"


def upload_files(file_paths):
    url = f"{BASE_URL}/upload"

    # Подготовка файлов для отправки
    files = [('files', (open(file_path, 'rb'))) for file_path in file_paths]

    # Отправка POST-запроса с файлами
    response = requests.post(url, files=files)

    # Закрытие файлов после отправки
    for file_path in file_paths:
        open(file_path, 'rb').close()

    # Вывод ответа от сервера
    print("Upload Response:", response.json())

def preprocess_data(filename):
    url = f"{BASE_URL}/preprocess"
    data = {'filename': filename}

    response = requests.post(url, json=data)

    # Вывод статуса ответа
    print("Response Status Code:", response.status_code)

    # Вывод текста ответа
    print("Response Text:", response.text)

    # Попытка декодировать JSON
    try:
        response_json = response.json()
        print("Preprocess Response:", response_json)
    except requests.exceptions.JSONDecodeError as e:
        print("JSON Decode Error:", str(e))


# if __name__ == "__main__":
#     file_paths = [r'C:\Users\skills\Downloads\Модуль 3\training_datasets\test_data.csv']  # Укажите пути к вашим файлам
#     upload_files(file_paths)


# if __name__ == "__main__":
#     filename = 'test_data.csv'
#     preprocess_data(filename)


url = 'http://127.0.0.1:5000/select_dataset'

# Тест 1: Выбор существующего набора данных
response = requests.post(url, json={'dataset_name': 'test_data.pkl'})
print("Test 1 - Select existing dataset:")
print("Status Code:", response.status_code)
print("Response:", response.json())

# # Тест 2: Выбор несуществующего набора данных
# response = requests.post(url, json={'dataset_name': 'dataset3'})
# print("\nTest 2 - Select non-existing dataset:")
# print("Status Code:", response.status_code)
# print("Response:", response.json())


# URL вашего Flask приложения
url = 'http://127.0.0.1:5000/select_model'

# Тест 1: Выбор существующей модели
response = requests.post(url, json={'model_name': 'model.pkl'})
print("Test 1 - Select existing model:")
print("Status Code:", response.status_code)
print("Response:", response.json())
