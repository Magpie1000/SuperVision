import tensorflow as tf
from Utils.utils import evaluate_vmaf_score
from PIL import Image  # Pillow module

# save output to file
def save_to_file(output_file, out):
    # [SSAFY] save image to output_file as raw yuv444
    u = tf.reshape(out[:,:,0], [-1])
    y = tf.reshape(out[:,:,1], [-1])
    v = tf.reshape(out[:,:,2], [-1])
    out_ = tf.concat([u, y, v], axis = 0)
    # convert to bytes
    tf.clip_by_value(out_, 0, 255)
    out_bytes = tf.cast(out_, tf.uint8)
    # save output
    with open(output_file, 'wb') as stream:
        print(output_file)
        for byte in out_bytes.numpy():
            stream.write(byte)
    


# Save list of outputs
def save_data(inputs, outputs, hr_input=None):
    # save file format
    h = outputs.shape[1]
    w = outputs.shape[2]

    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        input_file = 'Output/inp_' + str(w // 2) + 'x' + str(h // 2) + '_yuv444_' + str(i) + '.yuv'
        output_file = 'Output/out_' + str(w) + 'x' + str(h) + '_yuv444_' + str(i) + '.yuv'
        hr_file = 'Output/inp_hr_' + str(w) + 'x' + str(h) + '_yuv444_' + str(i) + '.yuv'

        save_to_file(input_file, inp)
        save_to_file(output_file, out)

        if hr_input is not None:
            save_to_file(hr_file, hr_input[i])
            evaluate_vmaf_score(hr_file, output_file, w, h)


# Show Input and Output
def analyze_output(inp, out):
    # [SSAFY] Get hr image with some non-ai filter and output image and compare
    lr_image =Image.fromarray(tf.cast(inp, tf.uint8).numpy(), 'YCbCr')
    lr_image.show(title='Input')
    interpolation = Image.LANCZOS  # interpolation = Image.BILINEAR
    hr_image = lr_image.resize((480, 640), resample = interpolation)
    hr_image.show(title='Lanczos filter')
    out_image =Image.fromarray(tf.cast(out, tf.uint8).numpy(), 'YCbCr')
    out_image.show(title='AI Super Resolution')


