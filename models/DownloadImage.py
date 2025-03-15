import os
import time
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

# Đường dẫn tới chromedriver
CHROMEDRIVER_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"  # Thay đổi theo đường dẫn của bạn
path ="Z:\\GarbageClassification\\dataset_raw"
# Tạo thư mục để lưu ảnh
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Tải ảnh từ URL
def download_image(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Đã tải: {save_path}")
    except Exception as e:
        print(f"Lỗi tải {url}: {e}")

# Tải hình ảnh từ Google Images
def download_images_from_google(keyword, num_images, output_dir):
    # Tạo thư mục lưu ảnh
    image_dir = os.path.join(output_dir, keyword.replace(" ", "_"))
    create_directory(image_dir)

    # Cấu hình Chrome Options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Chạy không hiển thị giao diện
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # Khởi tạo trình duyệt
    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Truy cập Google Images
        driver.get("https://www.google.com/imghp")
        time.sleep(2)  # Chờ trang tải

        # Tìm và nhập từ khóa
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)  # Chờ kết quả tải

        # Cuộn trang để tải thêm ảnh
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Chờ thêm ảnh tải
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Lấy danh sách ảnh
        images = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
        count = 0
        for img in images:
            if count >= num_images:
                break
            try:
                # Lấy URL ảnh
                img.click()  # Nhấp vào ảnh để mở chi tiết
                time.sleep(1)  # Chờ popup tải
                actual_image = driver.find_element(By.CSS_SELECTOR, "img.n3VNCb")
                img_url = actual_image.get_attribute("src")
                if img_url and "http" in img_url:
                    save_path = os.path.join(image_dir, f"{keyword.replace(' ', '_')}_{count}.jpg")
                    download_image(img_url, save_path)
                    count += 1
            except (NoSuchElementException, Exception) as e:
                print(f"Lỗi với ảnh: {e}")
                continue

        print(f"Đã tải {count} ảnh cho từ khóa: {keyword}")

    except Exception as e:
        print(f"Lỗi tổng quát: {e}")

    finally:
        driver.quit()

# Danh sách từ khóa
keywords = [
    "chai nhựa Vietnam", "túi nilon Vietnam",
    "giấy vụn Vietnam", "bao bì giấy Vietnam",
    "lon kim loại Vietnam", "đồ kim loại tái chế",
    "chai thủy tinh Vietnam", "ly thủy tinh tái chế",
    "vỏ rau củ Vietnam", "thức ăn thừa Vietnam",
    "rác y tế Vietnam", "pin cũ Vietnam"
]

# Tải ảnh
for keyword in keywords:
    download_images_from_google(keyword, num_images=200, output_dir="path")