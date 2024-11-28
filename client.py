import streamlit as st
import requests
import pandas as pd
import os

# Настройки API
API_URL = "http://127.0.0.1:5000"

# Путь к директории с файлами
DIRECTORY = "client_information"

@st.dialog('Информация о разработчике')
def dialog():
    response = requests.get(f"{API_URL}/info")
    if response.status_code == 200:
        info_info = response.json()
        st.write(info_info)

# Основной интерфейс
st.title("Предсказание моделью модуля 3")
st.sidebar.header("Меню")
menu_option = st.sidebar.selectbox("Выберите опцию", ["Справка о методах", "Загрузить файл", "Выбрать исходники", "Получить предикт", "Визуализация графика",  "Руководство пользователя"])


# Справка
if menu_option == "Справка о методах":
    st.subheader("Справка")
    response = requests.get(f"{API_URL}/methods")
    if response.status_code == 200:
        methods_info = response.json()
        st.write("Система имеет следующие возможности:")
        for method, description in methods_info.items():
            st.write(f"* {description}")
    else:
        st.error("Ошибка при загрузке справки.")

# Загрузка файла
elif menu_option == "Загрузить файл":
    st.subheader("Загрузка файлов")
    uploaded_files = st.file_uploader("Выберите файлы для загрузки", accept_multiple_files=True)
    if st.button("Загрузить"):
        for file in uploaded_files:
            files = {"files": (file.name, file.getvalue())}
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                st.success(f"Файл {file.name} загружен.")
            else:
                st.error(f"Ошибка загрузки файла {file.name}.")
    if st.button('Обработать'):
        for file in uploaded_files:
            files = {"filename": (file.name)}
            print(files)
            response = requests.post(f"{API_URL}/preprocess", json=files)
            if response.status_code == 200:
                st.success(f"Файл {file.name} обработан.")
            else:
                st.error(f"Ошибка обработки файла {file.name}.")


# Выбор модели и набора данных
elif menu_option == "Выбрать исходники":
    st.subheader("Выбор модели и данных")

    # Получаем доступные наборы данных
    response = requests.get(f"{API_URL}/datasets")
    if response.status_code == 200:
        datasets = response.json()
    else:
        st.error("Ошибка при получении списка данных.")
        datasets = []

    # Получаем доступные модели
    response = requests.get(f"{API_URL}/models")
    if response.status_code == 200:
        models = response.json()
    else:
        st.error("Ошибка при получении списка моделей.")
        models = []

    # Создаем выпадающий список для выбора набора данных
    dataset_name = st.selectbox("Выберите набор данных", datasets)

    if st.button("Выбрать набор данных"):
        response = requests.post(f"{API_URL}/select_dataset", json={"dataset_name": dataset_name})
        if response.status_code == 200:
            st.success("Набор данных выбран.")
        else:
            st.error("Ошибка выбора набора данных.")

    # Создаем выпадающий список для выбора модели
    model_name = st.selectbox("Выберите модель", models)

    if st.button("Выбрать модель"):
        response = requests.post(f"{API_URL}/select_model", json={"model_name": model_name})
        if response.status_code == 200:
            st.success("Модель выбрана.")
        else:
            st.error("Ошибка выбора модели.")

# Получение предсказания
elif menu_option == "Получить предикт":
    st.subheader("Получение предикта")
    if st.button("Сделать предсказание"):
        response = requests.get(f"{API_URL}/predict")
        if response.status_code == 200:
            predictions = response.json()
            st.write("Предсказания:", predictions)
        else:
            st.error("Ошибка при получении предсказания.")

# Визуализация графика
elif menu_option == "Визуализация графика":
    st.subheader("График предсказаний")
    if st.button("Показать график"):
        response = requests.get(f"{API_URL}/plot")
        if response.status_code == 200:
            st.success("График создан. Проверьте визуализацию.")
        else:
            st.error("Ошибка при создании графика.")


elif menu_option == "Руководство пользователя":
    st.subheader("Руководство пользователя-2")

    # Получаем список файлов из API
    response = requests.get(f"{API_URL}/files")
    if response.status_code == 200:
        files = response.json()
    else:
        st.error("Не удалось получить список файлов.")
        files = []

    # Предлагаем пользователю выбрать файл для загрузки
    selected_file = st.selectbox("Выберите файл для загрузки:", files)

    if selected_file:
        # Читаем содержимое выбранного файла
        file_response = requests.get(f"{API_URL}/files/{selected_file}")

        if file_response.status_code == 200:
            file_content = file_response.text
            # Отображаем содержимое файла
            st.text_area("Выбранное руководство: ", file_content, height=300)

            # Возможность загрузить файл
            st.download_button(
                label="Скачать файл",
                data=file_response.content,
                file_name=selected_file,
                mime="text/plain"
            )
        else:
            st.error("Не удалось загрузить файл.")


if st.button('Информация о разработчике'):
    dialog()
