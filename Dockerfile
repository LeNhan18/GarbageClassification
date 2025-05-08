FROM tensorflow/tensorflow:latest-gpu

# Cài đặt các thư viện cần thiết
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các package Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Tạo thư mục làm việc
WORKDIR /app

# Copy code vào container
COPY . .

# Chạy ứng dụng
CMD ["python", "models/combined_model.py"] 