#!/usr/bin/env python
# coding: utf-8

# In[24]:


import sys
import tensorflow as tf
from PIL import Image  # Pillow module
from model import make_model,SuperResolutionModel


MAX_PEL_VALUE=255

## model과 인풋이 들어오면 sr을 한 Image를 보냄
def get_output(model,image):
    input=tf.convert_to_tensor(image, tf.float32)
    out_ = model(input[tf.newaxis])
    out = tf.clip_by_value(out_, 0, MAX_PEL_VALUE)

    return Image.fromarray(tf.cast(out[0], tf.uint8).numpy())


def test(input_file):
    inp_lr = load_jpg(input_file, True)
    out_hr = get_output(model,inp_lr)
    return tf.cast(out_hr, tf.uint8).numpy()





