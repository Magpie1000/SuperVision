import tensorflow as tf

class SuperResolutionModel(tf.keras.models.Model):
    def __init__(self, w, h):
        super().__init__()
        # Model Layers
        self.conv1 = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(8, 3, strides=1, padding = 'same'),
                        tf.keras.layers.LeakyReLU(alpha=0.25)
                    ])
        self.conv2 = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(8, 3, strides=1, padding = 'same'),
                        tf.keras.layers.LeakyReLU(alpha=0.25)
                    ])
        self.conv3 = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(3, 3, strides=1, padding = 'same'),
                    ])
        
        # Vars
        self.W = w
        self.H = h
    
    # Compile Model as tensorflow graph
    def build_graph(self):
        input_shape = (1, self.W, self.H, 3)
        self.build(input_shape)

        # Test graph on zero-input
        inputs = tf.zeros(input_shape, dtype=tf.float32)
        self.call(inputs)

    # Main hypothesis function i.e model_output = call(model_input)
    def call(self, x):
        # Bias
        x_bi = tf.image.resize(x, (self.W * 2, self.H * 2), method='bilinear')

        # Conv Layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.image.resize(x, (self.W * 2, self.H * 2), method='bilinear')
        x = self.conv3(x)

        # Final output
        out = tf.add(x, x_bi)
        return out


def make_model(w, h):
    model = SuperResolutionModel(w, h)
    model.build_graph()
    model.summary()
    model.load_weights('model_weghts.h5')
    return model