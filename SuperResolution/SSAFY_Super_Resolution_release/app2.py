from flask import *
from flask_cors import CORS
from PIL import Image
from io import BytesIO, StringIO
from base64 import b64encode
from model import make_model,SuperResolutionModel
import super_resolution_normal
import io




model=make_model(320,240)
MAX_PEL_VALUE=255

app = Flask(__name__)
CORS(app)

@app.route('/normal_image',methods=['POST'])
def sr_normal_filter():
    print("@@@post image")
    f = request.files['image']
    mode=request.args['mode']
    print(mode)
    print(f)
    sr_image=super_resolution_normal.super_resolution_normal_filter(f,2,mode)
    buf = BytesIO()
    # sr_image.save(buf, format="JPEG", quality=100)
    sr_image.save(buf, format="PNG", quality=100)
    response = make_response(b64encode(buf.getvalue()))
    response.headers.set("Content-Type", "image/jpeg")
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0")
