from PL import Image
import os

def resize_image(input_dir,output_dir,size=(224,224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        in filename.endswith('.jpg', '.jpeg', 'png');
        try:
            img = Image.open(os.path.join(input_dir, filename))
            img_resized = img.resize(size)
            img_resized.save(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"Error processing {filename}: {e}")
resize_image('Z:\Han-leData','Z:\GarbageClassification\data\non_recyclable\trash')