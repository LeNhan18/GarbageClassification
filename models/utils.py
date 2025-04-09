# utils.py
# Các hàm hỗ trợ cho xử lý ảnh, đổi tên, resize, chuẩn hóa...

import os
from PIL import Image
import shutil

def rename_images_in_folder(folder_path, prefix="img", start_index=1):
    """
    Đổi tên tất cả ảnh trong thư mục thành prefix + số thứ tự (VD: plastic1.jpg)
    """
    index = start_index
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[-1]
            new_name = f"{prefix}{index}{ext}"
            os.rename(file_path, os.path.join(folder_path, new_name))
            index += 1
    print(f"✅ Đã đổi tên toàn bộ ảnh trong {folder_path}")

def resize_images_in_folder(folder_path, size=(150, 150)):
    """
    Resize tất cả ảnh trong thư mục về kích thước size
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                img = Image.open(file_path)
                img = img.resize(size)
                img.save(file_path)
            except Exception as e:
                print(f"⚠️ Lỗi với file {filename}: {e}")
    print(f"✅ Đã resize ảnh trong {folder_path} về {size}")

def prepare_all_subfolders(root_dir, rename_prefix="img", resize_size=(150, 150)):
    """
    Resize và rename toàn bộ ảnh trong các thư mục con của root_dir
    """
    for subdir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, subdir)
        if os.path.isdir(sub_path):
            resize_images_in_folder(sub_path, size=resize_size)
            rename_images_in_folder(sub_path, prefix=rename_prefix)

if __name__ == '__main__':
    # Test nhanh
    folder = input("Nhập đường dẫn thư mục ảnh: ")
    prepare_all_subfolders(folder)
