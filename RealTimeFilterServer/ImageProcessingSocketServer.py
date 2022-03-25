from flask import *
from flask_socketio import *
from PIL import Image
from inference import super_resolution
import io

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"
my_socket = SocketIO(app, cors_allowed_origins="*")


@my_socket.on("connect")
def welcome():
    print("socket connected")
    # load_super_resolution_model()


@my_socket.on("message")
def handle_message(data):
    # 소켓통신에서 blob으로 이미지를 넘기면 바이트배열이 넘어오므로 바이트 변환하여 이미지를 읽은 후 이 이미지를 가지고 super resolution 진행 후 해당 이미지를 다시 전송
    image = Image.open(io.BytesIO(data))
    # image.show()

    # call super resolution module

    # example
    hr_image = super_resolution(image)
    hr_image = Image.fromarray(hr_image)
    #hr_image.show()
    buf = io.BytesIO()
    hr_image.save(buf, format="JPEG")
    #image.save(buf, format="JPEG")
    send(buf.getvalue())


if __name__ == "__main__":
    my_socket.run(app, port=5000)
