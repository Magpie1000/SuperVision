from flask import *
import sys
import tensorflow as tf
from flask_socketio import *
from PIL import Image
from super_resolution_cnn import *
from model import make_model,SuperResolutionModel
import io

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"
my_socket = SocketIO(app, cors_allowed_origins="*")

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


@my_socket.on("message",namespace='/cnn')
def handle_message(data):
    # 소켓통신에서 blob으로 이미지를 넘기면 바이트배열이 넘어오므로 바이트 변환하여 이미지를 읽은 후 이 이미지를 가지고 super resolution 진행 후 해당 이미지를 다시 전송
    image = Image.open(io.BytesIO(data))
    # image.show()

    # call super resolution module

    # example
    hr_image = get_output(model,tf.convert_to_tensor(image, tf.float32))
    hr_image = Image.fromarray(tf.cast(hr_image, tf.uint8).numpy())
    #hr_image.show()
    buf = io.BytesIO()
    hr_image.save(buf, format="JPEG")
    #image.save(buf, format="JPEG")
    send(buf.getvalue())
    
@my_socket.on("message",namespace='/normal')
def handle_message(data):
    # 소켓통신에서 blob으로 이미지를 넘기면 바이트배열이 넘어오므로 바이트 변환하여 이미지를 읽은 후 이 이미지를 가지고 super resolution 진행 후 해당 이미지를 다시 전송
    image = Image.open(io.BytesIO(data))
    # image.show()

    # call super resolution module

    # example
    hr_image = image.resize((image.width*2,image.height*2),Image.NEAREST)
    # hr_image = Image.fromarray(tf.cast(hr_image, tf.uint8).numpy())
    #hr_image.show()
    buf = io.BytesIO()
    hr_image.save(buf, format="JPEG")
    #image.save(buf, format="JPEG")
    send(buf.getvalue())


if __name__ == "__main__":
    my_socket.run(app, port=5000)
