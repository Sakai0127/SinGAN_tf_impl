import tensorflow as tf

def conv_block(inputs, name='conv_block', layer_idx=0, out_channels=32, activation=tf.keras.layers.LeakyReLU(0.2), bn=True):
    x = tf.keras.layers.Conv2D(out_channels, 3, padding='SAME', name='%s_conv_%d'%(name, layer_idx))(inputs)
    if bn:
        x = tf.keras.layers.BatchNormalization(name='%s_BN_%d'%(name, layer_idx))(x)
    x = activation(x)
    return x

def G_block(inputs, noise, name='G_block', hidden_maps=32, num_layers=5, out_channel=3):
    with tf.name_scope(name):
        x = inputs + noise
        x = conv_block(x, name='conv_block_0', layer_idx=0, out_channels=hidden_maps)
        for i in range(1, num_layers-1, 1):
            x = conv_block(x, name='conv_block_%d'%i, layer_idx=i, out_channels=hidden_maps)
        x = conv_block(x, name='conv_block_%d'%(num_layers-1), layer_idx=num_layers-1, out_channels=out_channel, activation=tf.keras.activations.tanh, bn=False)
    return x + inputs

def D_block(inputs, name='D_block', hidden_maps=32, num_layers=5):
    with tf.name_scope(name):
        x = conv_block(inputs, name='conv_block_0', layer_idx=0, out_channels=hidden_maps)
        for i in range(1, num_layers-1, 1):
            x = conv_block(x, name='conv_block_%d'%i, layer_idx=i, out_channels=hidden_maps)
        x = conv_block(x, name='conv_block_%d'%(num_layers-1), layer_idx=num_layers-1, out_channels=1, activation=tf.keras.activations.linear, bn=False)
    return x