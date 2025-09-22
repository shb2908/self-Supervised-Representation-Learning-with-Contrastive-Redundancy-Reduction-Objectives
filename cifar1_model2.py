import os
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
import tensorflow as tf  # framework
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
tf.config.optimizer.set_jit(True)


class Augmentation(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    @tf.function
    def random_execute(self, prob: float) -> bool:
        return tf.random.uniform([], minval=0, maxval=1) < prob


class RandomToGrayscale(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.random_execute(0.2):
            x = tf.image.rgb_to_grayscale(x)
            x = tf.tile(x, [1, 1, 3])
        return x

class RandomColorJitter(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.random_execute(0.8):
            x = tf.image.random_brightness(x, 0.8)
            x = tf.image.random_contrast(x, 0.4, 1.6)
            x = tf.image.random_saturation(x, 0.4, 1.6)
            x = tf.image.random_hue(x, 0.2)
        return x


class RandomFlip(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.random_execute(0.5):
            x = tf.image.random_flip_left_right(x)
        return x


class RandomResizedCrop(Augmentation):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def call(self, x: tf.Tensor) -> tf.Tensor:
        rand_size = tf.random.uniform(
            shape=[],
            minval=int(0.75 * self.image_size),
            maxval=1 * self.image_size,
            dtype=tf.int32,
        )

        crop = tf.image.random_crop(x, (rand_size, rand_size, 3))
        crop_resize = tf.image.resize(crop, (self.image_size, self.image_size))
        return crop_resize


class RandomSolarize(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.random_execute(0.2):
            # flips abnormally low pixels to abnormally high pixels
            x = tf.where(x < 10, x, 255 - x)
        return x


class RandomBlur(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.random_execute(0.2):
            s = np.random.random()
            return tfa.image.gaussian_filter2d(image=x, sigma=s)
        return x


class RandomAugmentor(keras.Model):
    def __init__(self, image_size: int):
        super().__init__()

        self.image_size = image_size
        self.random_resized_crop = RandomResizedCrop(image_size)
        self.random_flip = RandomFlip()
        self.random_color_jitter = RandomColorJitter()
        self.random_blur = RandomBlur()
        self.random_to_grayscale = RandomToGrayscale()
        self.random_solarize = RandomSolarize()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.random_resized_crop(x)
        x = self.random_flip(x)
        x = self.random_color_jitter(x)
        x = self.random_blur(x)
        x = self.random_to_grayscale(x)
        x = self.random_solarize(x)

        x = tf.clip_by_value(x, 0, 1)
        return x


bt_augmentor = RandomAugmentor(IMAGE_SIZE)


class BTDatasetCreator:
    def __init__(self, augmentor: RandomAugmentor, seed: int = 1024):
        self.options = tf.data.Options()
        self.options.threading.max_intra_op_parallelism = 1
        self.seed = seed
        self.augmentor = augmentor

    def augmented_version(self, ds: list) -> tf.data.Dataset:
        return (
            tf.data.Dataset.from_tensor_slices(ds)
            .shuffle(1000, seed=self.seed)
            .map(self.augmentor, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            .with_options(self.options)
        )

    def __call__(self, ds: list) -> tf.data.Dataset:
        a1 = self.augmented_version(ds)
        a2 = self.augmented_version(ds)

        return tf.data.Dataset.zip((a1, a2)).with_options(self.options)

augment_versions = BTDatasetCreator(bt_augmentor)(train_features)


class BarlowLoss(keras.losses.Loss):
    def __init__(self, batch_size: int=BATCH_SIZE,**kwargs):
        super().__init__()
        self.lambda_amt = 5e-3
        self.batch_size = batch_size

    def get_off_diag(self, c: tf.Tensor) -> tf.Tensor:
        zero_diag = tf.zeros(c.shape[-1])
        return tf.linalg.set_diag(c, zero_diag)

    def cross_corr_matrix_loss(self, c: tf.Tensor) -> tf.Tensor:
        c_diff = tf.pow(tf.linalg.diag_part(c) - 1, 2)
        off_diag = tf.pow(self.get_off_diag(c), 2) * self.lambda_amt
        loss = tf.reduce_sum(c_diff) + tf.reduce_sum(off_diag)
        return loss

    def normalize(self, output: tf.Tensor) -> tf.Tensor:
        return (output - tf.reduce_mean(output, axis=0)) / (tf.math.reduce_std(
            output, axis=0
        )+1e-9)

    def cross_corr_matrix(self, z_a_norm: tf.Tensor, z_b_norm: tf.Tensor) -> tf.Tensor:
        return (tf.transpose(z_a_norm) @ z_b_norm) / self.batch_size

    def call(self, z_a: tf.Tensor, z_b: tf.Tensor) -> tf.Tensor:
        z_a_norm, z_b_norm = self.normalize(z_a), self.normalize(z_b)
        c = self.cross_corr_matrix(z_a_norm, z_b_norm)
        loss = self.cross_corr_matrix_loss(c)
        return loss
    

class WeightCoorLoss:
  def __init__(self):
    self.norm_l2 = lambda x,a : tf.nn.l2_normalize(x, axis=a)
    pass

  def __call__(self,w):
    """
    z1(a) -> z2(b)
    w (b,a)
    """
    w = tf.cast(w,tf.float32)
    dim1, dim2 = w.shape
    w = self.norm_l2(w,0)

    wcorr =  (tf.transpose(w) @ w)

    return tf.reduce_sum((tf.eye(dim2) - wcorr)**2)
  
class ResNet34:
    def identity_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation("relu")(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def convolutional_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filter, (3, 3), padding="same", strides=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation("relu")(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        # Processing Residue with conv(1,1)
        x_skip = tf.keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def __call__(self, shape=(32, 32, 3)):
        # Step 1 (Setup Input Layer)
        x_input = tf.keras.layers.Input(shape)
        x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
        # Step 2 (Initial Conv layer along with maxPool)
        x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
        # Define size of sub-blocks and initial filter size
        block_layers = [3, 4, 6, 3]
        filter_size = 64
        # Step 3 Add the Resnet Blocks
        for i in range(4):
            if i == 0:
                # For sub-block 1 Residual/Convolutional block not needed
                for j in range(block_layers[i]):
                    x = self.identity_block(x, filter_size)
            else:
                # One Residual/Convolutional Block followed by Identity blocks
                # The filter size will go on increasing by a factor of 2
                filter_size = filter_size * 2
                x = self.convolutional_block(x, filter_size)
                for j in range(block_layers[i] - 1):
                    x = self.identity_block(x, filter_size)
        # Step 4 End Dense Network
        x = tf.keras.layers.AveragePooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        model = tf.keras.models.Model(inputs=x_input, outputs=x, name="ResNet34")
        return model
    

def build_twin(n_dense_neurons = 5000,n_proj_layers = 2) -> keras.Model:
    # encoder network
    resnet = ResNet34()()
    last_layer = resnet.layers[-1].output

    # intermediate layers of the projector network
    outputs = [last_layer]
    for i in range(n_proj_layers):
        dense = tf.keras.layers.Dense(n_dense_neurons, name=f"projector_dense_{i}")
        if i == 0:
            x = dense(last_layer)
        else:
            x = dense(x)
        x = tf.keras.layers.BatchNormalization(name=f"projector_bn_{i}")(x)
        x = tf.keras.layers.ReLU(name=f"projector_relu_{i}")(x)
        outputs.append(x)

    x = tf.keras.layers.Dense(n_dense_neurons, name=f"projector_dense_{n_proj_layers}")(x)
    outputs.append(x)

    model = keras.Model(resnet.input, outputs)
    return model

class BarlowModel(keras.Model):
    def __init__(self,weight_coor_loss_obj):
        super().__init__()
        self.model = build_twin(n_dense_neurons = 512)
        self.model.build(input_shape=(None,IMAGE_SIZE,IMAGE_SIZE,3))
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.weight_coor_loss_obj = weight_coor_loss_obj

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, batch: tf.Tensor) -> tf.Tensor:
        # get the two augmentations from the batch
        x_a, x_b = batch

        with tf.GradientTape() as tape:
            # get two versions of predictions
            outputs_a, outputs_b = self.__call__(x_a, training=True), self.model(x_b, training=True)
            e0_a, p0_a,p1_a,p2_a = outputs_a
            e0_b, p0_b,p1_b,p2_b = outputs_b

            w23 = self.model.weights[-2]
            w12 = self.model.weights[-8]
            w01 = self.model.weights[-14]

            b0_loss = self.loss(e0_a, e0_b)
            b1_loss = self.loss(p0_a, p0_b)
            b2_loss = self.loss(p1_a, p1_b)
            b3_loss = self.loss(p2_a, p2_b)

            w23_loss = self.weight_coor_loss_obj(w23)
            w12_loss = self.weight_coor_loss_obj(w12)
            w01_loss = self.weight_coor_loss_obj(w01)

            loss = b3_loss + (w01_loss + w12_loss + w23_loss)


        grads_model = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads_model, self.model.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": loss,"b0_loss":b0_loss,"b1_loss":b1_loss,"b2_loss":b2_loss,"b3_loss":b3_loss,"w01_loss":w01_loss,"w12_loss":w12_loss,"w23_loss":w23_loss,"w01":w01,"w12":w12,"w23":w23}

    def call(self, x):
        return self.model(x) 
    
