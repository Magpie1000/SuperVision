import sys
import os
from PIL import Image  # Pillow module
import tensorflow as tf  # Tensorflow module

from datasaver import save_to_file
from Utils.utils import run_shell, get_file_list, does_not_exists
from Utils.options import parseArgs_data_processing

# Load JPG image
def load_jpg(filename, low_res=False):
    image = Image.open(filename)
    # interpolation = Image.LANCZOS  # default interpolation = Image.BICUBIC
    if low_res:
        image_vga = image.resize((240, 320))  # resample = interpolation
    else:
        image_vga = image.resize((480, 640))
    img_yuv = image_vga.convert("YCbCr")
    image.close()
    return tf.convert_to_tensor(img_yuv, tf.float32)


# Read raw yuv frame
def open_raw_yuv_frame(filename, w, h):  # Image is h x w
    def process_frame_channel(stream, w, h):
        raw_channel = stream.read(w * h)
        raw_channel_array = tf.convert_to_tensor(list(raw_channel), tf.float32)
        return tf.reshape(raw_channel_array, (w, h))

    def process_frame(stream, w, h):  # 444 format
        u = process_frame_channel(stream, w, h)
        y = process_frame_channel(stream, w, h)
        v = process_frame_channel(stream, w, h)
        return tf.stack([u, y, v], axis=-1)

    stream = open(filename, "rb")
    return process_frame(stream, w, h)



# Make a low resolution image using ffmpeg
def low_resolution(image):
    tempfile = "Output/temp_480x640_yuv444.yuv"
    tempfile_lr = "Output/temp_240x320_yuv444.yuv"

    # Save raw yuv
    save_to_file(tempfile, image)

    # Scale down image
    # [SSAFY] ffmpeg shell command to get low resolution image and save in tempfile_lr
    # scale_cmd = f'ffmpeg -i {tempfile} -vf scale=320:240 {tempfile_lr}'
    scale_cmd = f'ffmpeg -video_size 480x640 -pix_fmt yuv444p -i {tempfile} -vf scale="240:320" -pix_fmt yuv444p {tempfile_lr} -y'
    run_shell(scale_cmd)

    # Load LR image
    return open_raw_yuv_frame(tempfile_lr, 320, 240)


# Read all images from folder
def read_data(folder,filenames):
    if not folder.endswith("/"):
        folder = folder + "/"
    features_ = []
    targets_ = []
    print(len(filenames))
    for basename in filenames:
        image = load_jpg(folder + basename)
        image_lr = low_resolution(image)
        print(image.shape, image_lr.shape)
        features_.append(image_lr)
        targets_.append(image)
    features = tf.convert_to_tensor(features_, tf.float32)
    targets = tf.convert_to_tensor(targets_, tf.float32)
    print(features.shape, targets.shape)
    return features, targets


# Save loaded images as TFRecord
def save_dataset(dataset_folder,file_list, save_filename="Data/DataSet0.tfrecord"):
    features, targets = read_data(dataset_folder,file_list)
    writer = tf.io.TFRecordWriter(save_filename)
    for f, t in zip(features, targets):
        feature = {
            "lr_image": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(tf.cast(f, tf.uint8)).numpy()]
                )
            ),
            "hr_image": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[tf.io.serialize_tensor(tf.cast(t, tf.uint8)).numpy()]
                )
            ),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


if __name__ == "__main__":
    train, valid = parseArgs_data_processing(sys.argv)
    print(train, valid)
    filenames_train = get_file_list("../images/train")
    filenames_valid = get_file_list("../images/valid")
    cnt=len(filenames_train)//100;
    
    data_path="Data/"
    for i in range(1, cnt+2):   
        train_files=[]
        
        while filenames_train:
            train_files.append(filenames_train.pop())
            if len(train_files)>=100:
                break
        if len(train_files)>0:
            save_dataset(train, train_files,data_path+"DataSet_train"+str(i)+".tfrecord")
    
    cnt=len(filenames_valid)//100;
    

    for i in range(1,cnt+2):

        valid_files=[]
        
        while filenames_valid:
            valid_files.append(filenames_valid.pop())
            if len(valid_files)>=100:
                break

        if len(valid_files)>0:
            save_dataset(valid, valid_files,data_path+"DataSet_valid"+str(i)+".tfrecord")
        
            
    
            
    

        
        
    # save_dataset(train, "Data/train/")
    # save_dataset(valid, "Data/valid/")

