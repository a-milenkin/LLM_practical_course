# Используем базовый образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем зависимости проекта в контейнер
COPY requirements.txt .

# Устанавливаем зависимости проекта
RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в контейнер
COPY . .

# Открываем порт, на котором будет работать приложение Streamlit
EXPOSE 8501

# Запускаем приложение Streamlit
CMD ["streamlit", "run", "app.py"]