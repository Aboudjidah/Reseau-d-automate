#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 06:22:39 2022

@author: idriss
"""

import numpy as np
from random import shuffle
import tensorflow as tf
import sys
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
def new_conv_layer(input,num_input_channels,filter_size,\
num_filters,use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1],\
padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],\
strides=[1, 2, 2, 1],padding='SAME')
        layer = tf.nn.relu(layer)
    return layer, weights
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features
def new_fc_layer(input,num_inputs,num_outputs,use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer
def creer_model():
    global saver,x,x_image,y_true,y_true_cls,layer_conv1, weights_conv1,\
layer_conv2, weights_conv2,layer_flat, num_features,layer_fc1,layer_fc2,\
y_pred,y_pred_cls,cross_entropy,cost,optimizer,correct_prediction,accuracy,session
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)
    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,num_input_channels=num_channels,filter_size=filter_size1,num_filters=num_filters1,use_pooling=True)
    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,num_input_channels=num_filters1,filter_size=filter_size2,num_filters=num_filters2,use_pooling=True)
    layer_flat, num_features = flatten_layer(layer_conv2)
    layer_fc1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size,use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc1,num_inputs=fc_size,num_outputs=num_classes,use_relu=False)

    y_pred = tf.nn.softmax(layer_fc2)

    y_pred_cls = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,labels=y_true)

    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    saver=tf.train.Saver()

def optimize(num_iterations):

    print("apprentissage en cours pour",num_iterations,"Iterations...")



    for i in range(num_iterations//5):
        print(" ",end="")

    print("|100%|")


    for i in range(num_iterations):

        ind=0

        indice_aleatoire = list(range(len(dessins)))

        shuffle(indice_aleatoire)

        dessins_apprend=[]

        vects_apprend=[]

        vetcs=np.array(vects)

        for i in indice_aleatoire:

            dessins_apprend.append(dessins[i])

            vects_apprend.append(vects[i])

        for t in range(len(dessins)//train_batch_size):

            x_batch, y_true_batch = \
np.array(dessins_apprend[ind:ind+train_batch_size]),\
np.array(vects_apprend[ind:ind+train_batch_size])

            ind+=train_batch_size

            feed_dict_train = {x: x_batch,y_true: y_true_batch}

            session.run(optimizer, feed_dict=feed_dict_train)

        if i%5 ==0:

            sys.stdout.write("#")

            sys.stdout.flush()
def recuperer():

    global noms,dessins,association,nbrlabels,vects
    fichier = open("dataperso.txt", 'r')
    texte = fichier.read()

    try:

        texte2=texte.split(",")

        texte2.remove(texte2[-1])

        nbras=int(texte2[0])

        nbrlabels=int(texte2[1])

        association = texte2[2:nbras+2]

        noms = texte2[nbras+2:nbras+nbrlabels+2]

        texte3=texte2[nbras+nbrlabels+2:]

        i=0

        for z in range(nbras):

            lis=[]

            for y in range(784):

                try:

                    lis.append(float(texte3[i]))
                except:
                    lis.append(0.0)
                    i+=1
                    dessins.append(lis)
                    vects=[]
                for name in association:
                    vect=[]
                    for a in range(nbrlabels):
                        if noms.index(name)==a:
                            vect.append(1)
                        else:
                            vect.append(0)
                            vects.append(vect)
                            vects=np.array(vects)
    except:
        print("fichier vide")
        fichier.close()
def print_accuracy():
    global dessins_faux
    nbr_juste=0
    dessins_faux=[]
    for i in range(len(dessins)):
        img=dessins[i]
        ass=association[i]
        pred = session.run(y_pred_cls,feed_dict={ x : [img]})
        if (pred+1)%10==int(ass):
            nbr_juste+=1
        else:
            dessins_faux.append(img)
    print("taux sur le learning set:",(nbr_juste/len(dessins))*100,"%")
dessins=[]
vects=[]
recuperer()
print(len(dessins))
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10
filter_size1 = 5
num_filters1 = 16
filter_size2 = 5
num_filters2 = 36
fc_size = 128
train_batch_size = 80
xt=dessins
yt = vects
taille_input=np.size(xt[0])
taille_output=np.size(yt[0])
creer_model()
try:
    saver.restore(session, "weights_entrainepasMNIST2/model.ckpt")
    print("model restauré")
except:
    pass
    optimize(num_iterations=5)
    save_path = saver.save(session, "weights_entrainepasMNIST2/model.ckpt")
print_accuracy()