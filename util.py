import numpy as np
from PIL import Image
import tensorflow as tf

def gradient_penalty(real, fake, netD):
    with tf.name_scope('gradient_penalty'):
        e = tf.random.normal([1])
        x_hat = real*e + (1-e)*fake
        with tf.GradientTape() as gp:
            gp.watch(x_hat)
            d_hat = netD(x_hat)
        grad = gp.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.nn.l2_loss(grad))
        p = tf.reduce_mean((ddx - 1.0)**2)
    return p

def calc_g_loss(real, in_z, noise, netD, netG, alpha):
    fake = netG([in_z, noise])
    fake_d = netD(fake)
    with tf.name_scope('calc_g_loss'):
        g_loss_adv = -tf.reduce_mean(fake_d, axis=[1, 2, 3])
        rec_loss = alpha * tf.reduce_mean(tf.losses.mean_squared_error(real, fake))
        g_loss = g_loss_adv + rec_loss
    return g_loss

def calc_d_loss(real, in_z, noise, netD, netG, alpha):
    fake = netG([in_z, noise])
    fake_d = netD(fake)
    real_d = netD(real)
    with tf.name_scope('calc_d_loss'):
        d_loss_adv = -tf.reduce_mean(real_d, axis=[1, 2, 3]) + tf.reduce_mean(fake_d, axis=[1, 2, 3])
        gp = gradient_penalty(real, fake, netD)
        d_loss = d_loss_adv + gp
    return d_loss

def load_image(path):
    img = np.array(Image.open(path), dtype=np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, 0)
    return img
