#!/usr/bin/env python
# coding: utf-8

# In[24]:


import sys
import tensorflow as tf
from PIL import Image  # Pillow module
from model import make_model,SuperResolutionModel


model=make_model(320,240)

MAX_PEL_VALUE=255

## model과 인풋이 들어오면 sr을 한 numpy 배열 보냄
def get_output(model,input):
    out_ = model(input[tf.newaxis])
    out = tf.clip_by_value(out_, 0, MAX_PEL_VALUE)

    return out[0]


def test(input_file):
    inp_lr = load_jpg(input_file, True)
    out_hr = get_output(model,inp_lr)
    return tf.cast(out_hr, tf.uint8).numpy()





