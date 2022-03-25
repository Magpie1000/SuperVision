from flask import *
from flask_cors import CORS
from PIL import Image
from inference import super_resolution
from io import BytesIO, StringIO
from base64 import b64encode

app = Flask(__name__)
CORS(app)

@app.route("/image", methods=["POST"])
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

if __name__ == "__main__":
    app.run(host="0.0.0.0")

    