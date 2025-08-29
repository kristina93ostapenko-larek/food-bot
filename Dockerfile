FROM python:3.11-slim

WORKDIR /app

# Копируем только нужные файлы
COPY bot.py .
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Запускаем бота
CMD ["python", "bot.py"]
