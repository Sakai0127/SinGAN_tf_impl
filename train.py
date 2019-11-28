import argparse
from collections import OrderedDict
import math
import os

from PIL import Image
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import model
#from SinGAN import model
import util
#from SinGAN import util

def train_SR(args):
    real_image = util.load_image(args.input_image)
    real_image = tf.constant(real_image)
    _, h, w, _ = real_image.shape
    scale_factor = math.pow(1/2, 1/3)
    n_blocks = int(math.log(args.min_size / min(h, w), scale_factor))+1
    save_dir = '%s/%s'%(args.save_dir, args.input_image.split('.')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    resolutions = []
    Gs = []
    Ds = []
    

    for i in range(n_blocks):
        scale = math.pow(scale_factor, n_blocks-i-1)
        cur_h, cur_w = int(h*scale), int(w*scale)
        img = tf.image.resize(real_image, (cur_h, cur_w))
        resolutions.append((cur_h, cur_w))
        inp = tf.keras.Input(shape=(None, None, 3))
        noise = tf.keras.Input(shape=(None, None, 3))
        G = tf.keras.Model(inputs=[inp, noise], outputs=model.G_block(inp, noise, name='G_block_%d'%i, hidden_maps=args.hidden_channels, num_layers=args.n_layers))
        D = tf.keras.Model(inputs=inp, outputs=model.D_block(inp, name='D_block_%d'%i, hidden_maps=args.hidden_channels, num_layers=args.n_layers))
        lr_g = tf.Variable(args.lr_g, trainable=False)
        lr_d = tf.Variable(args.lr_d, trainable=False)
        opt_G = tf.keras.optimizers.Adam(lr_g, args.beta1)
        opt_D = tf.keras.optimizers.Adam(lr_d, args.beta1)
        if i > 0:
            for (prev, cur) in zip(Gs[-1].layers, G.layers):
                cur.set_weights(prev.get_weights())
            for (prev, cur) in zip(Ds[-1].layers, D.layers):
                cur.set_weights(prev.get_weights())
            init_opt(opt_G, G)
            init_opt(opt_D, D)
        with tqdm(range(args.n_iter)) as bar:
            bar.set_description('Block %d / %d'%(i+1, n_blocks))
            for iteration in bar:
                if i == 0:
                    prev_img = tf.zeros_like(img)
                else:
                    prev_img = proc_image(tf.zeros([1, resolutions[0][0], resolutions[0][1], 3]), Gs, args.noise_weight, resolutions)
                g_loss, d_loss = train_step(img, prev_img, args.noise_weight, G, D, opt_G, opt_D, args.g_times, args.d_times, args.alpha)
                bar.set_postfix(ordered_dict=OrderedDict(
                    g_loss=g_loss.numpy(), d_loss=d_loss.numpy()
                ))
                if iteration == int(args.n_iter*0.8):
                    lr_d.assign(args.lr_d*0.1)
                    lr_g.assign(args.lr_g*0.1)
        Gs.append(G)
        Ds.append(D)
        G.save(os.path.join(save_dir, 'SR_G_%d_res_%dx%d.h5'%(i+1, cur_h, cur_w)))
        D.save(os.path.join(save_dir, 'SR_D_%d_res_%dx%d.h5'%(i+1, cur_h, cur_w)))

    scale_factor = math.pow(1/2, 1/3)
    target_res = 4
    scale = 1.0 / scale_factor
    n, h, w, c = real_image.shape
    t_h, t_w = h*target_res, w*target_res
    iter_times = int(math.log(target_res, scale))
    img = real_image
    os.makedirs(os.path.join(save_dir, 'result'), exist_ok=True)
    for i in range(1, iter_times+1, 1):
        res = (int(h*math.pow(scale, i)), int(w*math.pow(scale, i)))
        img = tf.image.resize(img, size=res)
        img = G([img, tf.zeros_like(img)])
        image = np.squeeze(img)
        image = (np.clip(image, -1.0, 1.0) + 1.0) * 127.5
        image = Image.fromarray(image.astype(np.uint8))
        image.save('SinGAN/models/sample/result/%dx%d.jpg'%res)

def init_opt(opt, model):
    opt.iterations
    opt._create_hypers()
    opt._create_slots(model.trainable_weights)

@tf.function
def proc_image(img, Gs, noise_w, res):
    for (G, r) in zip(Gs, res[1:]):
        noise = tf.random.normal(tf.shape(img))*noise_w
        img = tf.image.resize(G([img, noise]), r)
    return img

@tf.function
def train_step(real, g_input, noise_w, G, D, opt_G, opt_D, g_times, d_times, alpha):
    for _ in range(d_times):
        with tf.GradientTape() as d_tape:
            noise = tf.random.normal(tf.shape(real))*noise_w
            d_loss = util.calc_d_loss(real, g_input, noise, D, G, alpha)
        d_grad = d_tape.gradient(d_loss, D.trainable_weights)
        opt_D.apply_gradients(zip(d_grad, D.trainable_weights))
    for _ in range(g_times):
        with tf.GradientTape() as g_tape:
            noise = tf.random.normal(tf.shape(real))*noise_w
            g_loss = util.calc_g_loss(real, g_input, noise, D, G, alpha)
        g_grad = g_tape.gradient(g_loss, G.trainable_weights)
        opt_G.apply_gradients(zip(g_grad, G.trainable_weights))
    return g_loss, d_loss