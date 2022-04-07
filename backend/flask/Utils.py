import os
import subprocess
import shlex
import sys
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
import redis
from tensorflow.python.saved_model import tag_constants
from xml.etree.ElementTree import parse
from io import BytesIO
from base64 import b64decode, b64encode
from PIL import Image
from super_resolution_normal import super_resolution_normal_filter
from inference import super_resolution
from uuid import uuid4


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
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == b"" and process.poll() is not None:
            break
        if output:
            print(output.strip())
    process.poll()
    tree = parse("output.xml")
    pooled_metrics = tree.find("pooled_metrics")
    metric = pooled_metrics.find("metric[15]")
    vmaf_score = metric.attrib["min"]
    os.remove("output.xml")
    return vmaf_score

def get_upscaled_images_with_vmaf_scores(original_image):
    lanczos_upscaled = super_resolution_normal_filter(original_image, mode="NEAREST", x=4)
    sr_upscaled = super_resolution(original_image)
    sr_upscaled = Image.fromarray(sr_upscaled)
    
    # save to yuv444 format each Images
    width, height = original_image.size
    lr_image = original_image.resize((int(width/2), int(height/2)))
    lr_lanczos_upscaled = super_resolution_normal_filter(lr_image, mode="NEAREST", x=2)
    lr_sr_upscaled = super_resolution(lr_image)
    lr_sr_upscaled = Image.fromarray(lr_sr_upscaled)

    lr_lanczos_upscaled = lr_lanczos_upscaled.resize((width, height))
    lr_sr_upscaled = lr_sr_upscaled.resize((width, height))

    original_file_name = str(uuid4()) + ".yuv"
    sr_file_name = str(uuid4()) + ".yuv"
    normal_file_name = str(uuid4()) + ".yuv"
    save_to_yuv(original_file_name, convert_jpg_to_yuv_tensor(original_image))
    save_to_yuv(sr_file_name, convert_jpg_to_yuv_tensor(lr_sr_upscaled))
    save_to_yuv(normal_file_name, convert_jpg_to_yuv_tensor(lr_lanczos_upscaled))
    # get VMAF score

    normal_vmaf_score = get_vmaf_score(normal_file_name, original_file_name, width, height)
    sr_vmaf_score = get_vmaf_score(sr_file_name, original_file_name, width, height)

    os.remove(original_file_name)
    os.remove(sr_file_name)
    os.remove(normal_file_name)

    sr_buf = BytesIO()
    normal_buf = BytesIO()

    sr_upscaled.save(sr_buf, "JPEG")
    lanczos_upscaled.save(normal_buf, "JPEG")

    # return responese with upscaled images and VMAF result
    return {
        "sr_upscaled" : b64encode(sr_buf.getvalue()).decode("utf-8"),
        "normal_upscaled" : b64encode(normal_buf.getvalue()).decode("utf-8"),
        "sr_vmaf_score" : sr_vmaf_score,
        "normal_vmaf_score" : normal_vmaf_score
    }


def detect(original_image):
    image = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (416, 416))
    image = image / 255.

    images = []
    for i in range(1):
        images.append(image)
    images = np.asarray(images).astype(np.float32)

    saved_model_loaded = tf.saved_model.load("./weights/yolov4-tiny-416", tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures["serving_default"]
    batch_data = tf.constant(images)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    return get_object_coordinations_with_class(original_image, pred_bbox)

def read_class_names(class_file_name):
    names ={}
    with open(class_file_name, "r") as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip("\n")
    return names

def get_object_coordinations_with_class(image, bboxes, classes=read_class_names("./data/classes/coco.names")):
    num_classes = len(classes)
    w, h = image.size
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    result = []
    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * h)
        coor[2] = int(coor[2] * h)
        coor[1] = int(coor[1] * w)
        coor[3] = int(coor[3] * w)

        score = out_scores[0][i]
        class_index = int(out_classes[0][i])
        info = {
            "class" : classes[class_index], 
            "y1" : str(coor[0]), 
            "x1" : str(coor[1]), 
            "y2" : str(coor[2]), 
            "x2" : str(coor[3]), 
            "score" : str(score)
            }
        result.append(info)
    
    return result


def save_at_cache(image):
    try:
        con = redis.StrictRedis(host="localhost",port=6379,db=2)
        buffer = BytesIO()
        image.save(buffer, "JPEG")
        con.set("image", b64encode(buffer.getvalue()), 3600) # expire cache after 1hour

    except Exception as e:
        print(e)
        
def load_from_cache(key):
    try:
        con = redis.StrictRedis(host="localhost",port=6379,db=2)
        image_bytes = con.get(key)
        return Image.open(BytesIO(b64decode(image_bytes)))
    except Exception as e:
        print(e)
