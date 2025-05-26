import os
import hashlib
from PIL import Image
import imagehash
import shutil
from datetime import datetime

# --- Cấu hình ---
data_dir = 'Z:\\GarbageClassification\\datas\\non_recyclable'  # Thư mục chứa dữ liệu
output_log_dir = 'Z:\\GarbageClassification\\logs'  # Thư mục lưu log
backup_dir = 'Z:\\GarbageClassification\\backup\\trash'  # Thư mục lưu bản sao trước khi xóa
similarity_threshold = 10  # Ngưỡng cho perceptual hash (tăng lên 10)


def get_file_hash(file_path):
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Lỗi khi tính hash cho {file_path}: {e}")
        return None
def get_perceptual_hash(file_path):
    try:
        img = Image.open(file_path)
        return imagehash.difference_hash(img)  # Sử dụng difference hash thay vì average hash
    except Exception as e:
        print(f"Lỗi khi tính perceptual hash cho {file_path}: {e}")
        return None
def find_and_remove_duplicates(data_dir, backup_dir, output_log_dir):
    # Tạo thư mục backup và log
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(output_log_dir, exist_ok=True)

    # Tạo file log với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_log_dir, f'duplicate_removal_log_{timestamp}.txt')

    # Thu thập tất cả file ảnh
    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    print(f"Tìm thấy {len(image_files)} ảnh trong thư mục.")
#Hash anh timf kiem so sanh anh khac nhau
    # Lưu trữ hash và đường dẫn
    hash_dict = {}
    duplicates = []

    for file_path in image_files:
        img_hash = get_file_hash(file_path)
        if img_hash is None:
            continue

        if img_hash in hash_dict:
            duplicates.append((file_path, hash_dict[img_hash]))
        else:
            hash_dict[img_hash] = file_path

    # Ghi log và xử lý ảnh trùng lặp
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Duplicate Removal Log - {timestamp}\n")
        f.write(f"Total images scanned: {len(image_files)}\n")
        f.write(f"Duplicates found: {len(duplicates)}\n\n")

        if duplicates:
            print(f"🗑Tìm thấy {len(duplicates)} ảnh trùng lặp. Bắt đầu xử lý...")
            for dup, orig in duplicates:
                # Sao lưu ảnh trùng lặp vào backup_dir
                backup_path = os.path.join(backup_dir, os.path.relpath(dup, data_dir))
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copy(dup, backup_path)

                # Xóa ảnh trùng lặp
                try:
                    os.remove(dup)
                    log_message = f"Removed: {dup} (Original: {orig})\n"
                    print(log_message.strip())
                except Exception as e:
                    log_message = f"Error removing {dup}: {e}\n"
                    print(f"{log_message.strip()}")

                f.write(log_message)
        else:
            print("Không tìm thấy ảnh trùng lặp.")
            f.write("No duplicates found.\n")

    return len(duplicates)


# --- Kiểm tra ảnh tương tự (không xóa, chỉ liệt kê) ---
def find_similar_images(data_dir, output_log_dir, threshold=10):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_log_dir, f'similar_images_log_{timestamp}.txt')

    image_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    print(f"📸 Tìm thấy {len(image_files)} ảnh để kiểm tra tương tự.")

    hash_dict = {}
    similar_pairs = []

    for file_path in image_files:
        img_hash = get_perceptual_hash(file_path)
        if img_hash is None:
            continue

        for existing_hash, existing_path in hash_dict.items():
            # Tính khoảng cách Hamming
            hamming_distance = img_hash - existing_hash  # Difference hash hỗ trợ trừ trực tiếp
            if hamming_distance <= threshold:
                similar_pairs.append((file_path, existing_path, hamming_distance))

        hash_dict[img_hash] = file_path

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Similar Images Log - {timestamp}\n")
        f.write(f"Total images scanned: {len(image_files)}\n")
        f.write(f"Similarity threshold: {threshold}\n")
        f.write(f"Similar pairs found: {len(similar_pairs)}\n\n")

        if similar_pairs:
            for file1, file2, distance in similar_pairs:
                log_message = f"Similar: {file1} and {file2} (Hamming distance: {distance})\n"
                f.write(log_message)
        else:
            f.write("No similar images found.\n")

    print(f" Tìm thấy {len(similar_pairs)} cặp ảnh tương tự. Kiểm tra log tại {log_file}.")
    return similar_pairs


# --- Chạy chương trình ---
if __name__ == "__main__":
    print(" Bắt đầu kiểm tra và loại bỏ ảnh trùng lặp...")

    # Bước 1: Xóa ảnh giống hệt (MD5 hash)
    num_duplicates = find_and_remove_duplicates(data_dir, backup_dir, output_log_dir)

    # Bước 2: Kiểm tra ảnh tương tự (không xóa)
    print("\n Kiểm tra ảnh tương tự...")
    similar_pairs = find_similar_images(data_dir, output_log_dir, similarity_threshold)

    print(f"\n Hoàn thành! Tìm thấy và xóa {num_duplicates} ảnh trùng lặp.")
    print(f" Log được lưu tại: {output_log_dir}")
    print(f" Bản sao ảnh trùng lặp được lưu tại: {backup_dir}")