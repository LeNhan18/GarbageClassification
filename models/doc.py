import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Cài đặt các tham số
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Tiền xử lý dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Thay đổi kích thước ảnh để phù hợp với mô hình
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
])

# Tải dữ liệu
train_dataset = datasets.ImageFolder(root='path/to/train_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root='path/to/test_data', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Sử dụng mô hình pre-trained ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Số lớp đầu ra tương ứng với số loại rác

# Di chuyển mô hình tới GPU (nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Định nghĩa loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass và cập nhật trọng số
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Kiểm tra mô hình trên bộ dữ liệu kiểm tra
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
