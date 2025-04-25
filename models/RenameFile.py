import os
paths = "D:\\nl\\archive\\Garbage classification\\Garbage classification\\cardboard"
start =1
#Đổi tên file
for i ,filename in enumerate(sorted(os.listdir(paths))):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        new_name =f"cardboard{start + i}.jpg"
        old_path =os.path.join(paths,filename)
        new_path = os.path.join(paths,new_name)
        os.rename(old_path,new_path)