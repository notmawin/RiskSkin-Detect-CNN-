FROM python:3.9

# ลง Library สำหรับ OpenCV (ที่ทำให้เกิด Error libGL บน Render)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Hugging Face Spaces บังคับให้ใช้ Port 7860
CMD ["python", "app.py"]
