import os

# === Đường dẫn thư mục dataset ===
data_dir = r"Z:\\GarbageClassification\\data"

def count_images(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                count += 1
    return count

def main():
    class_names = ['recyclable', 'non_recyclable']
    total_images = 0
    class_counts = {}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"⚠️ Không tìm thấy thư mục: {class_name}")
            continue

        count = count_images(class_path)
        class_counts[class_name] = count
        total_images += count

    print("\n📊 Kết quả rà soát dữ liệu:")
    print(f"Tổng ảnh: {total_images}")
    for class_name in class_names:
        count = class_counts.get(class_name, 0)
        percent = (count / total_images) * 100 if total_images > 0 else 0
        warn = "⚠️" if percent < 40 else ""
        print(f"{class_name:<18}: {count:>6} ảnh  ({percent:.2f}%) {warn}")

    if abs(class_counts['recyclable'] - class_counts['non_recyclable']) > 0.3 * total_images:
        print("\n⚠️ DỮ LIỆU MẤT CÂN BẰNG NHIỀU → NÊN xử lý (augmentation hoặc dùng class_weight)")

if __name__ == '__main__':
    main()
