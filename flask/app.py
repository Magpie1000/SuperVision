from argparse import Namespace
from operator import mod
from flask import *
from flask_cors import CORS
from importlib_metadata import files
import tensorflow as tf
from flask_socketio import *
from PIL import Image
from super_resolution_normal import super_resolution_normal_filter
from super_resolution_cnn import *
from base64 import b64encode, b64decode
from model import make_model,SuperResolutionModel
from io import BytesIO, StringIO
import super_resolution_normal
from inference import super_resolution
from uuid import uuid4
from Utils import convert_jpg_to_yuv_tensor, save_to_yuv, get_vmaf_score
from os import remove

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"
my_socket = SocketIO(app, cors_allowed_origins="*")
CORS(app)
model=make_model(320,240)
MAX_PEL_VALUE=255

@my_socket.on("connect",namespace='/cnn')
def welcome():
    print("cnn socket connected")
    # load_super_resolution_model()
    
@my_socket.on("connect",namespace='/normal')
def welcome():
    print("normal socket connected")
    # load_super_resolution_model()
    
    

@my_socket.on("connect")
def welcome():
    print("gan socket connected")
    # load_super_resolution_model()



@my_socket.on("message",namespace='/gan')
def handle_message(data):
    # 소켓통신에서 blob으로 이미지를 넘기면 바이트배열이 넘어오므로 바이트 변환하여 이미지를 읽은 후 이 이미지를 가지고 super resolution 진행 후 해당 이미지를 다시 전송
    image = Image.open(BytesIO(data))
    # image.show()

    # call super resolution module

    # example
    hr_image = super_resolution(image)
    hr_image = Image.fromarray(hr_image)
    #hr_image.show()
    buf = BytesIO()
    hr_image.save(buf, format="JPEG")
    #image.save(buf, format="JPEG")
    send(buf.getvalue())


@my_socket.on("message",namespace='/cnn')
def handle_message(data):
    # 소켓통신에서 blob으로 이미지를 넘기면 바이트배열이 넘어오므로 바이트 변환하여 이미지를 읽은 후 이 이미지를 가지고 super resolution 진행 후 해당 이미지를 다시 전송
    image = Image.open(BytesIO(data))
    # image.show()
    # call super resolution module
    # example
    hr_image = get_output(model,image)
    hr_image = Image.fromarray(tf.cast(hr_image, tf.uint8).numpy())
    #hr_image.show()
    buf = BytesIO()
    hr_image.save(buf, format="JPEG")
    #image.save(buf, format="JPEG")
    send(buf.getvalue())
    
@my_socket.on("message",namespace='/normal')
def handle_message(data):
    # 소켓통신에서 blob으로 이미지를 넘기면 바이트배열이 넘어오므로 바이트 변환하여 이미지를 읽은 후 이 이미지를 가지고 super resolution 진행 후 해당 이미지를 다시 전송
    image = Image.open(BytesIO(data))
    # image.show()

    # call super resolution module

    # example
    hr_image = image.resize((image.width*2,image.height*2),Image.NEAREST)
    # hr_image = Image.fromarray(tf.cast(hr_image, tf.uint8).numpy())
    #hr_image.show()
    buf = BytesIO()
    hr_image.save(buf, format="JPEG")
    #image.save(buf, format="JPEG")
    send(buf.getvalue())

@app.route("/image", methods=["POST"])
def handle_image_request():
    # upscale with AI Super Resolution filter and Lanczos Interpollation
    original_image = request.files["image"]
    original_image = Image.open(original_image)
    lanczos_upscaled = super_resolution_normal_filter(original_image, mode="LANCZOS", x=4)
    sr_upscaled = super_resolution(original_image)
    sr_upscaled = Image.fromarray(sr_upscaled)
    
    # save to yuv444 format each Images
    width, height = original_image.size
    lr_image = original_image.resize((int(width/4), int(height/4)))
    lr_lanczos_upscaled = super_resolution_normal_filter(lr_image, mode="LANCZOS", x=4)
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

    remove(original_file_name)
    remove(sr_file_name)
    remove(normal_file_name)

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
    


@app.route("/image/gan", methods=["POST"])
def image_process():
    original_image = request.files["image"]
    
    original_image = Image.open(original_image)
    # original_image.show()
    hr_image = super_resolution(original_image)
    hr_image = Image.fromarray(hr_image)
    # hr_image.show()
    buf = BytesIO()
    hr_image.save(buf, format="JPEG", quality=100)
    response = make_response(b64encode(buf.getvalue()))
    response.headers.set("Content-Type", "image/jpeg")
    return response
    
@app.route('/image/normal',methods=['POST'])
def sr_normal_filter():
    print("@@@post image")
    input_image = request.files['image']
    mode=request.form['mode']
    rate=float(request.form['rate'])
    hr_image=super_resolution_normal.super_resolution_normal_filter(input_image,rate,mode)
    buf = BytesIO()
    # sr_image.save(buf, format="JPEG", quality=100)
    hr_image.save(buf, format="PNG", quality=100)
    response = make_response(b64encode(buf.getvalue()))
    response.headers.set("Content-Type", "image/jpeg")
    return response

@app.route('/image/cnn',methods=['POST'])
def sr_cnn_filter():
    print("@@@cnn post image")
    input_image = request.files['image']
    image = Image.open(input_image)
    
    #cnn 필터는 3 채널이여서 png 파일을 RGB로 변환해서 넣어줘야함
    if 'png' in str(input_image).lower():
        print("PNG")
        image.convert('RGB')
    elif 'jpg' or 'jpeg' in str(input_image).lower():
        print("JPG")
    else:
        print("not image file")
        return "BAD REQUEST"

    hr_image = get_output(model,image)
    hr_image = Image.fromarray(tf.cast(hr_image, tf.uint8).numpy())

    buf = BytesIO()
    hr_image.save(buf, format="JPEG", quality=100)
    # hr_image.save(buf, format="PNG", quality=100)
    response = make_response(b64encode(buf.getvalue()))
    response.headers.set("Content-Type", "image/jpeg")
    return response


if __name__ == "__main__":
    my_socket.run(app, host="0.0.0.0", port=5000)
