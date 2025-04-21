import os
from collections import defaultdict

# === Cấu hình thư mục dataset ===
data_dir = r"Z:\\GarbageClassification\\data\\recyclable"  # Thay đường dẫn nếu cần

def count_images_per_class(base_dir):
    class_counts = defaultdict(int)
    total_images = 0

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(image_files)
        class_counts[class_name] = count
        total_images += count

    return class_counts, total_images

def analyze_class_distribution(class_counts, total_images):
    print(f"\n📊 Tổng số ảnh: {total_images}")
    print(f"{'Class':<25} {'Số ảnh':<10} {'Tỷ lệ (%)':<10}")
    print("-" * 50)

    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        ratio = (count / total_images) * 100
        warning = "⚠️" if ratio < 10 else ""
        print(f"{class_name:<25} {count:<10} {ratio:>6.2f}% {warning}")

    print("\n⚠️ Cảnh báo nếu class < 10% tổng số ảnh → Có thể bị model học lệch.")

if __name__ == "__main__":
    class_counts, total_images = count_images_per_class(data_dir)
    analyze_class_distribution(class_counts, total_images)
