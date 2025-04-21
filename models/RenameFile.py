import os
paths = "Z:\\GarbageClassification\\data\\non_recyclable\\battery"

start =1
#Đổi tên file
for i ,filename in enumerate(sorted(os.listdir(paths))):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        new_name =f"battery{start + i}.jpg"
        old_path =os.path.join(paths,filename)
        new_path = os.path.join(paths,new_name)
        os.rename(old_path,new_path)