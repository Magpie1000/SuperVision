import math

from PIL import Image  # Pillow module
import tensorflow as tf
from model import make_model
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class SuperResolutionTrainer:
    def __init__(self, model, model_path):
        # Model
        self.model = model

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # Loss Function
        self.loss = tf.keras.losses.MeanSquaredError()

        # Metric Function
        self.metric_loss = tf.keras.metrics.Mean()

        # Support Variables
        self.MAX_PEL_VALUE = 255
        self.model_path = model_path

    # Training Function
    def train(self, train_input, train_target, epochs, batch_size):
        n_input = train_input.shape[0]
        num_steps = 1 + ((n_input - 1) // batch_size)

        # Loop for Epochs
        shuffle_index_array = tf.range(n_input)
        print(train_input, train_target)
        for cur_epoch in range(epochs):
            self.metric_loss.reset_states()

            # Loop for batch
            tf.random.shuffle(shuffle_index_array)
            for step_index in range(num_steps):
                # [SSAFY] get random sample lr, hr batch images ex dim lr = (batch_size,320,240,3)
                lr = train_input[shuffle_index_array[step_index]]
                hr = train_target[shuffle_index_array[step_index]]
                loss, _ = self.train_step(cur_epoch, lr, hr)
                self.metric_loss(loss)

            # Metric Result
            print("Epoch: {}, loss: {}".format(cur_epoch, self.metric_loss.result()))

            if cur_epoch > 0 and cur_epoch % 50 == 0:
                self.model.save(self.model_path, save_format='tf')
        self.model.save_weights("model_weghts.h5",overwrite=True)
    
    # Training step on batch size (Gradient optimization)
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, n_epoch, lr, hr):
        # Add steps to Gradient Tape
        with tf.GradientTape() as tape:
            out_ = self.model(tf.expand_dims(lr, axis=0))
            out = tf.clip_by_value(out_, 0, self.MAX_PEL_VALUE)
            _out = out / self.MAX_PEL_VALUE
            _hr = hr / self.MAX_PEL_VALUE
            loss_value = self.loss(_hr, _out)

        gradients = tape.gradient(loss_value, self.model.trainable_variables)

        # Apply optimizer
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss_value, out_

    # Test Function on single input
    def output(self, input):
        out_ = self.model(input[tf.newaxis])
        out = tf.clip_by_value(out_, 0, self.MAX_PEL_VALUE)
        # hr_image = Image.fromarray(tf.cast(out[0], tf.uint8).numpy())
        # hr_image.show()
        return out

    # Evaluate Function
    def evaluate(self, test_input, test_target, batch_size=1, save_image=False):
        self.metric_loss.reset_states()
        psnr_values = []
        outputs = []
        for lr, hr in zip(test_input, test_target):
            out = self.output(lr)
            _out = out / self.MAX_PEL_VALUE
            _hr = hr[tf.newaxis] / self.MAX_PEL_VALUE

            loss_value = self.loss(_hr, _out)
            self.metric_loss(loss_value)

            # [SSAFY] get psnr loss value
            psnr_value = tf.image.psnr(_hr, _out, max_val=1.0)
            # psnr_value = tf.reduce_mean(tf.boolean_mask(psnr_value, tf.math.is_finite(psnr_value)))
            psnr_values.append(psnr_value)
            print(f"psnr_value: {psnr_value}")
            if save_image:
                outputs.append(out)

        loss = self.metric_loss.result()
        psnr_loss = tf.reduce_mean(psnr_values).numpy()
        print("loss: {}, psnr_loss: {}".format(loss, psnr_loss))

        if save_image:
            # [SSAFY] return joint output
            return tf.concat(outputs, axis=0)


def get_trainer(w, h, model_path='Model/model_0', load_model=False):
    if load_model:
        model = tf.keras.models.load_model(model_path)
    else:
        model = make_model(w, h)
    trainer = SuperResolutionTrainer(model, model_path)
    return trainer
