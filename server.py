from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import time
from scipy.io import wavfile
from scipy import signal
import cv2
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
from catboost import CatBoostClassifier

app = Flask(__name__)

# Хранилище для загруженных файлов и моделей
preprocessed_datasets = {}
current_dataset = None
current_model = None
predictions = None
i = 0


###ПРОВЕРИЛА, РАБОТАЕТ
###КОММЕНТАРИИ С 3 РЕШЕТКАМИ ДЛЯ МЕНЯ, ИНАЧЕ ЗАБУДУ, ЧТО ТОЧНО ТЕСТИРОВАЛА
@app.route('/upload', methods=['POST'])
def upload_data():
    uploaded_files = []
    files = request.files.getlist('files')
    for file in files:
        file.save(os.path.join('uploads', file.filename))
        uploaded_files.append(file.filename)
    return jsonify(uploaded_files)

### Я ПРОКЛЯЛА МИР, НО ОНО ЗАРАБОТАЛО
@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    global preprocessed_datasets
    # список загруженных
    uploaded_files = [f for f in os.listdir('uploads') if os.path.isfile(os.path.join('uploads', f))]
    #имя для предобработки
    filename = request.json.get('filename')
    if filename not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    df = pd.read_csv(os.path.join('uploads', filename))
    df['feature_1'].replace(',', '.', inplace=True)
    df['feature_1'] = pd.to_numeric(df['feature_1'], errors='coerce')
    df['feature_2'].replace(',', '.', inplace=True)
    df['feature_2'] = pd.to_numeric(df['feature_2'], errors='coerce')
    df['feature_3'].replace(',', '.', inplace=True)
    df['feature_3'] = pd.to_numeric(df['feature_3'], errors='coerce')
    df.fillna(0)

    scaler = StandardScaler()
    processed_data = scaler.fit_transform(df)
    #отрезаем .csv
    filename = filename[:-4] + '.pkl'

    # Сохранение в формате pickle
    with open(os.path.join('processed', filename), 'wb') as f:
        pickle.dump(processed_data, f)
    memory_usage = processed_data.nbytes
    return jsonify({'message': 'Data preprocessed successfully', 'memory_usage': memory_usage})


###ТОЖЕ ВРОДЕ РАБОТАЕТ
@app.route('/datasets', methods=['GET'])
def list_datasets():
    # Извлекаем список файлов из папки processed в нужном формате
    try:
        datasets_files = [f for f in os.listdir('processed') if os.path.isfile(os.path.join('processed', f))]
        datasets_files = [f for f in datasets_files if f.endswith('.pkl')]
        return jsonify(datasets_files)
    except Exception as e:
        return jsonify({'error': 'Could not retrieve datasets', 'message': str(e)}), 500


### УРА, РАБОТАЕТ!!!!!
@app.route('/select_dataset', methods=['POST'])
def select_dataset():
    global current_dataset
    #список файлов диры обработанных исходников
    list = [f for f in os.listdir('processed') if os.path.isfile(os.path.join('processed', f))]
    #имя исходного датасета
    dataset_name = request.json.get('dataset_name')
    # сверяем
    if dataset_name not in list:
        return jsonify({'error': 'Dataset not found'}), 404
    current_dataset = dataset_name
    return jsonify({'message': f'Dataset {dataset_name} selected'})


### ТОЖЕ ЧТО_ТО ДАЕТ
@app.route('/models', methods=['GET'])
def list_models():
    # Извлекаем список файлов из папки models и только pkl
    try:
        model_files = [f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))]
        model_files = [f for f in model_files if f.endswith('.pkl')]
        return jsonify(model_files)
    except Exception as e:
        return jsonify({'error': 'Could not retrieve models', 'message': str(e)}), 500


### РАБОТАЕТ
@app.route('/select_model', methods=['POST'])
def select_model():
    global current_model
    #хватаем все файлы в папке
    model_files = [f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))]
    #получаем название модели
    model_name = request.json.get('model_name')
    #сверяем
    if model_name not in model_files:
        return jsonify({'error': 'Model not found'}), 404
    current_model = model_name
    return jsonify({'message': f'Model {model_name} selected'})


@app.route('/predict', methods=['GET'])
def make_predictions():
    global predictions
    if current_dataset is None or current_model is None:
        return jsonify({'error': 'No dataset or model selected'}), 400
    print(current_model)
    print(current_model)
    #собираем пути
    work_model = os.path.join('models', current_model)
    print(work_model)
    work_dataset = os.path.join('processed', current_dataset)
    print(work_dataset)
    #читаем
    with open (work_model, 'rb') as file:
        model = pickle.load(file)
    df = pd.read_pickle(work_dataset)
    predictions = model.predict(df)
    return predictions.tolist()


@app.route('/plot', methods=['GET'])
def plot_predictions():
    global predictions
    import matplotlib.pyplot as plt

    # Получаем список чисел из вложенных списков
    numbers = [point[0] for point in predictions]

    # Создаем график
    plt.plot(numbers, 'bo')

    # Устанавливаем подпись оси X
    plt.xlabel('Индекс точки')

    # Устанавливаем подпись оси Y
    plt.ylabel('Значение точки')

    # Показываем график
    plt.show()
    methods_info = {
        'result': 'On screen'
    }
    return jsonify(methods_info)



### ТОЖЕ ФУНЦИОНИРУЕТ ХУДО-БЕДНО
@app.route('/methods', methods=['GET'])
def api_methods():
    methods_info = {
        'upload': 'Загрузка файлов на сервер',
        'preprocess': 'Обработка загруженных файлов',
        'datasets': 'Вывод списка доступных датасетов',
        'select_dataset': 'Выбор датасета для работы',
        'models': 'Вывод списка доступных моделей',
        'select_model': 'Выбор модели для работы',
        'predict': 'Предсказание моделью на выбранных данных',
        'plot': 'Визуализация графика',
        'methods': 'Справка',
        'version': 'Информация о разработчике'
    }
    return jsonify(methods_info)

### НЕ ПАДАЕТ И ХОРОШО
@app.route('/info', methods=['GET'])
def api_version():
    version_info = {
        'Версия продукта': '1.0.1',
        'Разработчик': 'Адамова Александра',
        'Организация': 'ИГЭУ'
    }
    return jsonify(version_info)


@app.route('/files', methods=['GET'])
def list_files():

    FILE_DIRECTORY = "client_information"
    try:
        # Получаем список файлов в директории
        files = os.listdir(FILE_DIRECTORY)
        files = [f for f in files if os.path.isfile(os.path.join(FILE_DIRECTORY, f))]
        return jsonify(files)
    except Exception as e:
        # Обработка исключений
        return jsonify({'error': str(e)}), 500

@app.route('/files/<filename>', methods=['GET'])
def get_file(filename):
    DIRECTORY = "client_information"
    try:
        if os.path.isfile(os.path.join(DIRECTORY, filename)):
            return send_from_directory(DIRECTORY, filename)
        else:
            return jsonify({'error': 'file not found'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Обработка ошибок
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad Request', 'message': str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not Found', 'message': str(error)}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error', 'message': str(error)}), 500

def fit_model (path):
    #считывание файла
    df = pd.read_csv(path)
    X = df.drop('label', axis = 1)
    y = df['label']
    # Нормализация данных
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations = 1000, depth = 5, learning_rate = 0.1)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models\\CatBoostClassifier.pkl')

###СЕРВЕР РАБОТАЕТ
if __name__ == '__main__':

    # Создание необходимых директорий, если они не существуют
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    if i==0:
        fit_model(r'C:\Users\skills\Downloads\Модуль 3\training_datasets\train_data.csv.')
        i=1

    app.run(debug=True)
