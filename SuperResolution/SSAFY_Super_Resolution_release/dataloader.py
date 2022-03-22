from fileinput import filename
import tensorflow as tf
from Utils.utils import run_shell, get_file_list, does_not_exists
from Utils.utils import does_not_exists


# Parse Example
def parse_dataset_example(example):
    features = tf.io.parse_single_example(example,
                                          features={
                                              'lr_image': tf.io.FixedLenFeature([], tf.string),
                                              'hr_image': tf.io.FixedLenFeature([], tf.string)
                                          })

    lr = tf.io.parse_tensor(features['lr_image'], out_type=tf.uint8)
    hr = tf.io.parse_tensor(features['hr_image'], out_type=tf.uint8)
    return tf.cast(lr, tf.float32), tf.cast(hr, tf.float32), 0


# Load TFRecord Dataset
def load_dataset(dataset_path='Data/'):
    train_valid='train'
    if 'valid' in dataset_path:
        train_valid='valid'
        
    filenames = get_file_list("Data/")
    dataset_path='Data/'
    features_ = []
    targets_ = []
    image_sz = []
    
    for name in filenames:
        if train_valid in name:
            data_path=dataset_path+name
            dataset = tf.data.TFRecordDataset(data_path, compression_type='', buffer_size=256 << 20)
            for elem in dataset:
                f, t, sz = parse_dataset_example(elem)
                features_.append(f)
                targets_.append(t)
                image_sz.append(sz)
            

    features = tf.convert_to_tensor(features_, tf.float32)
    targets = tf.convert_to_tensor(targets_, tf.float32)
    print(features.shape, targets.shape)
    return features, targets, image_sz
