import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


batch_size = 32
learning_rate =0.001
num_epochs = 10

#Tiền xử lý dữ liệu
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Tải dữ liệu
dulieu_hoc =datasets.ImageFolder(root='#',transform=transform)
dulieu_tai = DataLoader(dulieu_hoc,batch_size= batch_size,shuffle=True)

dulieu_ktra = datasets.ImageFolder(root='#',transform=transform)
dulieu_tai_ktra = DataLoader(dulieu_ktra,batch_size=batch_size, shuffle=True)

#Sử dụng pretrained ResNet50
model = models.resnet50(pretrained=True)
#Đóng băng trọng số của các lớp trước
for param in model.parameters():
    param.requires_grad = False
# Thya đổi lớp phân loại cuối cùng
model.fc = nn.Linear(model.fc.in_features, 2) # Sử dụng 2 lớp đầu ra (mô hình phân loại 2 lớp)
#Di chuyển mô hình đến GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Sử dụng hàm cross entropy loss(Hàm mất mát)
criterion = nn.CrossEntropyLoss()
#Sử dụng hàm Adam optimizer (Bộ tối ưu)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Tiến hành huấn luyện
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dulieu_tai):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass và tính đạo hàm
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Tính gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dulieu_tai)}')

# Kiểm tra độ chính xác
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dulieu_tai_ktra:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Độ chính xác : {100*correct/total:.2f}%")