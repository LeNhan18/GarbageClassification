FROM tensorflow/tensorflow:2.15.0

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Cài đặt các thư viện cần thiết
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements trước để tận dụng cache
COPY requirements.txt .

# Cài đặt các package Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy code vào container
COPY . .

# Tạo volume cho dữ liệu
VOLUME ["/app/datas"]

# Chạy ứng dụng
CMD ["python", "models/combined_model.py"] 