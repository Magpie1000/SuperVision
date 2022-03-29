from concurrent.futures import process
import os
import sys
import tensorflow as tf
from xml.etree.ElementTree import parse

def convert_jpg_to_yuv_tensor(image):
    return tf.convert_to_tensor(image.convert("YCbCr"), tf.float32)

def save_to_yuv(output_file, out):
    # [SSAFY] save image to output_file as raw yuv444
    u = tf.reshape(out[:, :, 0], [-1])
    y = tf.reshape(out[:, :, 1], [-1])
    v = tf.reshape(out[:, :, 2], [-1])
    out_ = tf.concat([u, y, v], axis=0)
    # convert to bytes
    tf.clip_by_value(out_, 0, 255)
    out_bytes = tf.cast(out_, tf.uint8)
    # save output
    with open(output_file, "wb") as stream:
        print(output_file)
        for byte in out_bytes.numpy():
            stream.write(byte)

def get_vmaf_score(sr_path, normal_path, width, height):
    if sys.platform == "linux" or sys.platform == "linux2":
        vmaf_path = "./vmaf"
    else:
        vmaf_path = "vmaf.exe"
    cmd = f"{vmaf_path} -r {normal_path} -d {sr_path} -w {width} -h {height} -p 444 -b 8 --feature psnr -o output.xml"
    return run_shell(cmd)
    
def run_shell(cmd):
    os.system(cmd)
    tree = parse("output.xml")
    pooled_metrics = tree.find("pooled_metrics")
    metric = pooled_metrics.find("metric[15]")
    vmaf_score = metric.attrib["min"]
    os.remove("output.xml")
    return vmaf_score