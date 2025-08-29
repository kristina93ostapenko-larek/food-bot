FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install aiogram==3.7.0
RUN pip install openai==1.3.9
RUN pip install python-dotenv==0.19.0
RUN pip install aiohttp==3.8.4

COPY . .

CMD ["python", "bot.py"]
