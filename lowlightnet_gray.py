import tensorflow as tf
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import read_csv
from PIL import Image
from random import randint

from library.Autoencoder import Autoencoder

def loadcsv(data):
    raw_data = read_csv(data)
    return raw_data.values[:,1:]

target = "dark"
path = "model" + target + "_gray" + datetime.datetime.now().strftime("%b_%d_%Y_%Hh_%Mm_%Ss")
# Read database
datasize = 356000
n_input = 289
# hyper parameters
n_samples = datasize
training_epochs = 30
batch_size = 2000
display_step = 1

corruption_level = 0
sparse_reg = 0.000001

#
n_inputs = n_input
n_hidden = 1000
n_hidden2 = 800
n_hidden3 = 500
n_outputs = n_input
lr = 0.1

# define the autoencoder
ae = Autoencoder(n_layers=[n_inputs, n_hidden],
                          transfer_function = tf.nn.sigmoid,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para = [corruption_level, sparse_reg],
                          reference = True)
ae_2nd = Autoencoder(n_layers=[n_hidden, n_hidden2],
                          transfer_function = tf.nn.sigmoid,
                          optimizer = tf.train.AdamOptimizer(learning_rate = lr),
                          ae_para=[corruption_level, sparse_reg],
                          reference = False)
ae_3rd = Autoencoder(n_layers=[n_hidden2, n_hidden3],
                          transfer_function = tf.nn.sigmoid,
                          optimizer = tf.train.AdamOptimizer(learning_rate = 0.1*lr),
                          ae_para=[corruption_level, sparse_reg],
                          reference = False)


## define the output layer using softmax in the fine tuning step
corrupt = tf.placeholder(tf.float32, [None, n_inputs])
h = corrupt

# Go through the three autoencoders
for layer in range(len(ae.n_layers) - 1):
    h = ae.transfer(
        tf.add(tf.matmul(h, ae.weights['encode'][layer]['w']),ae.weights['encode'][layer]['b']))
#for layer in range(len(ae_2nd.n_layers) - 1):
#    h = ae_2nd.transfer(
#        tf.add(tf.matmul(h, ae_2nd.weights['encode'][layer]['w']),ae_2nd.weights['encode'][layer]['b']))
#for layer in range(len(ae_3rd.n_layers) - 1):
#    h = ae_3rd.transfer(
#        tf.add(tf.matmul(h, ae_3rd.weights['encode'][layer]['w']),ae_3rd.weights['encode'][layer]['b']))
#for layer in range(len(ae_3rd.n_layers) - 1):
#    h = ae_3rd.transfer(
#        tf.add(tf.matmul(h, ae_3rd.weights['recon'][layer]['w']),ae_3rd.weights['recon'][layer]['b']))
#for layer in range(len(ae_2nd.n_layers) - 1):
#    h = ae_2nd.transfer(
#        tf.add(tf.matmul(h, ae_2nd.weights['recon'][layer]['w']),ae_2nd.weights['recon'][layer]['b']))
for layer in range(len(ae.n_layers) - 1):
    h = ae.transfer(
        tf.add(tf.matmul(h, ae.weights['recon'][layer]['w']),ae.weights['recon'][layer]['b']))

ref_input = tf.placeholder(tf.float32, [None, n_outputs]) #-> original value
denoised = h # -> predicted value

denoise_error = tf.reduce_sum(tf.pow(tf.subtract(ref_input, denoised), 2.0))

train_step  = tf.train.AdamOptimizer(learning_rate = lr).minimize(denoise_error)
train_step_after  = tf.train.AdamOptimizer(learning_rate = 0.1*lr).minimize(denoise_error)


## Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver(max_to_keep=None)
ckpt_path = os.path.join(path,"model")
ckpt = tf.train.get_checkpoint_state(path)

restore = False
savedpath = "modeldark_grayJan_26_2019_14h_28m_00s"
saved_ckpt_path = os.path.join(savedpath,"model")
saved_ckpt = tf.train.get_checkpoint_state(savedpath)

if saved_ckpt and saved_ckpt.model_checkpoint_path and restore:
    saver.restore(sess, saved_ckpt.model_checkpoint_path)    
    print("Model  " + savedpath + "  Load Complete")
else:
    

    orig_data=np.empty((datasize,n_input))
    if target == "dark" :
        dark_data=np.empty((datasize,n_input))
    elif target == "noise" :
        nois_data=np.empty((datasize,n_input))
    elif target == "combine" :
        comb_data=np.empty((datasize,n_input))
    else :
        print("no matching model")
        exit(1)
    #orig_data=np.empty((datasize,n_input))
    #dark_data=np.empty((datasize,n_input))
    #nois_data=np.empty((datasize,n_input))
    #comb_data=np.empty((datasize,n_input))

    try:
        orig_data = loadcsv("gray_original.csv")
        print("orig_data load complete")
        if target == "dark" :
            dark_data = loadcsv("gray_dark.csv")
            print("dark_data load complete")
        elif target == "noise" :
            nois_data = loadcsv("gray_noise.csv")
            print("nois_data load complete")
        elif target == "combine" :
            comb_data = loadcsv("gray_combine.csv")
            print("comb_data load complete")
        else :
            dark_data = loadcsv("gray_dark.csv")
            print("dark_data load complete")
            nois_data = loadcsv("gray_noise.csv")
            print("nois_data load complete")
            comb_data = loadcsv("gray_combine.csv")
            print("comb_data load complete")
    except :
        for i in range (datasize):
            if i%1000 == 0:
                print(i)
            filename_o = 'original/' + str(201+i) + '.jpg';
            filename_d = 'darken/' + str(201+i) + '.jpg';
            filename_n = 'noise/' + str(201+i) + '.jpg';
            filename_c = 'combine/' + str(201+i) + '.jpg';

            img_o = Image.open( filename_o ).convert('L')
            img_d = Image.open( filename_d ).convert('L')
            img_n = Image.open( filename_n ).convert('L')
            img_c = Image.open( filename_c ).convert('L')

            try:
                temp_o = np.asarray( img_o, dtype='uint8' )
            except SystemError:
                temp_o = np.asarray( img_o.getdata(), dtype='uint8' )

            try:
                temp_d = np.asarray( img_d, dtype='uint8' )
            except SystemError:
                temp_d = np.asarray( img_d.getdata(), dtype='uint8' )

            try:
                temp_n = np.asarray( img_n, dtype='uint8' )
            except SystemError:
                temp_n = np.asarray( img_n.getdata(), dtype='uint8' )
        #
            try:
                temp_c = np.asarray( img_c, dtype='uint8' )
            except SystemError:
                temp_c = np.asarray( img_c.getdata(), dtype='uint8' )

            temp_o = temp_o[:,:].ravel()
            temp_d = temp_d[:,:].ravel()
            temp_n = temp_n[:,:].ravel()
            temp_c = temp_c[:,:].ravel()


            orig_data[i,:] = np.true_divide(temp_o,255.);
            dark_data[i,:] = np.true_divide(temp_d,255.);
            nois_data[i,:] = np.true_divide(temp_n,255.);
            comb_data[i,:] = np.true_divide(temp_c,255.);


    print("Data Load Complete\n")

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            seed=randint(1,20000)
            random.seed = seed
            orig =  random.sample(orig_data,batch_size)
            if target == "dark":
                random.seed = seed
                batch = random.sample(dark_data,batch_size)
            elif target =="noise":
                random.seed = seed
                batch = random.sample(nois_data,batch_size)
            elif target == "combine":
                random.seed = seed
                batch = random.sample(comb_data,batch_size);
            else:
                print("no matching model name!")
                break;

            # Fit training using batch data
            temp = ae.partial_fit()
            cost, opt = sess.run(temp,feed_dict={ae.x: batch, ae.orig: orig, ae.keep_prob : ae.in_keep_prob})

            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch ae1:", '%d,' % (epoch + 1),
                "Cost:", "{:.9f}".format(avg_cost))

    print("************************First AE training finished******************************")


    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            seed=randint(1,20000)
            random.seed = seed
            orig =  random.sample(orig_data,batch_size)
            if target == "dark":
                random.seed = seed
                batch = random.sample(dark_data,batch_size)
            elif target =="noise":
                random.seed = seed
                batch = random.sample(nois_data,batch_size)
            elif target == "combine":
                random.seed = seed
                batch = random.sample(comb_data,batch_size);
            else:
                print("no matching model name!")
                break;

            # Fit training using batch data
            h_ae1_out = sess.run(ae.transform(),feed_dict={ae.x: batch, ae.keep_prob : ae.in_keep_prob})
            h_ae1_ref = sess.run(ae.transform(),feed_dict={ae.x: orig, ae.keep_prob : ae.in_keep_prob})
            temp = ae_2nd.partial_fit()
            cost, opt = sess.run(temp,feed_dict={ae_2nd.x: h_ae1_out, ae_2nd.orig: h_ae1_ref, ae_2nd.keep_prob : ae_2nd.in_keep_prob})

            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch ae2:", '%d,' % (epoch + 1),
                "Cost:", "{:.9f}".format(avg_cost))

    print("************************Second AE training finished******************************")

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # Fit training using batch data
            seed=randint(1,20000)
            random.seed = seed
            orig =  random.sample(orig_data,batch_size)
            if target == "dark":
                random.seed = seed
                batch = random.sample(dark_data,batch_size)
            elif target =="noise":
                random.seed = seed
                batch = random.sample(nois_data,batch_size)
            elif target == "combine":
                random.seed = seed
                batch = random.sample(comb_data,batch_size);
            else:
                print("no matching model name!")
                break;
            h_ae1_out = sess.run(ae.transform(),feed_dict={ae.x: batch, ae.keep_prob : ae.in_keep_prob})
            h_ae2_out = sess.run(ae_2nd.transform(),feed_dict={ae_2nd.x: h_ae1_out, ae_2nd.keep_prob : ae_2nd.in_keep_prob})

            h_ae1_ref = sess.run(ae.transform(),feed_dict={ae.x: orig, ae.keep_prob : ae.in_keep_prob})
            h_ae2_ref = sess.run(ae_2nd.transform(),feed_dict={ae_2nd.x: h_ae1_ref, ae_2nd.keep_prob : ae_2nd.in_keep_prob})

            temp = ae_3rd.partial_fit()
            cost, opt = sess.run(temp,feed_dict={ae_3rd.x: h_ae2_out, ae_3rd.orig: h_ae2_ref, ae_3rd.keep_prob : ae_3rd.in_keep_prob})

            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch ae3:", '%d,' % (epoch + 1),
                "Cost:", "{:.9f}".format(avg_cost))

    print("************************Third AE training finished******************************")


    # Training the softmax layer
    for epoch in range(200):
        cost = 0.
        seed=randint(1,20000)
        random.seed = seed
        orig =  random.sample(orig_data,batch_size)
        if target == "dark":
            random.seed = seed
            batch = random.sample(dark_data,batch_size)
        elif target =="noise":
            random.seed = seed
            batch = random.sample(nois_data,batch_size)
        elif target == "combine":
            random.seed = seed
            batch = random.sample(comb_data,batch_size);
        else:
            print("no matching model name!")
            break;
        _,cost = sess.run([train_step,denoise_error], feed_dict={corrupt: batch, ref_input: orig,
                                        ae.keep_prob: 1.0, ae_2nd.keep_prob: 1.0, ae_3rd.keep_prob: 1.0})
        if epoch % display_step == 0:
            print("Epoch finetune 1:", '%d,' % (epoch + 1),
                "Cost:", "{:.9f}".format(cost))
    print("*************************Finish the finetuning step1*****************************")


    # Training of fine tune

    best_cost = 9999999999999
    max_epoch = 1000
    epoch = 0
    #for epoch in range(max_epoch):
    while True:
        cost = 0.
        epoch +=1
        seed=randint(1,20000)
        random.seed = seed
        orig =  random.sample(orig_data,batch_size)
        if target == "dark":
            random.seed = seed
            batch = random.sample(dark_data,batch_size)
        elif target =="noise":
            random.seed = seed
            batch = random.sample(nois_data,batch_size)
        elif target == "combine":
            random.seed = seed
            batch = random.sample(comb_data,batch_size);
        else:
            print("no matching model name!")
            break;    
        _,cost = sess.run([train_step_after,denoise_error],feed_dict={corrupt: batch, ref_input: orig,
                                        ae.keep_prob: 1.0, ae_2nd.keep_prob: 1.0, ae_3rd.keep_prob: 1.0})
        if epoch % display_step == 0:
            print("Epoch finetune 2:", '%d,' % (epoch + 1),
                "Cost:", "{:.9f}".format(cost/10))
        else: None

        if best_cost > cost:
            saver.save(sess,ckpt_path, global_step=max_epoch)
            print("model  " + path + "  saved")
            best_cost = cost
            earlystop = 0
        else: 
            earlystop += 1
            if earlystop > 200:
                break
            else:None

    print("************************Finish the fine tuning******************************")
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess, ckpt.model_checkpoint_path)
# Test trained model

orig_test = loadcsv("gray_original_test.csv")
print("orig_test load complete")
if target == "dark" :
    test = loadcsv("gray_dark_test.csv")
    print("dark_data load complete")
elif target == "noise" :
    test = loadcsv("gray_noise_test.csv")
    print("nois_data load complete")
elif target == "combine" :
    test = loadcsv("gray_combine_test.csv")
    print("comb_data load complete")
else: None

modified = sess.run(denoised,
               feed_dict={corrupt:test})
mod_l1 = sess.run(ae.reconstruction,
                feed_dict={ae.x:test,ae.keep_prob:1.0})
for i in range(8370) :   
    arr = np.multiply(mod_l1[i,:],255.).reshape(17,17);
    if i%1000 == 0:
        print(i)
        print(arr.astype(np.uint8))
    else: None
    img = Image.fromarray(arr.astype(np.uint8), 'L')
    #img.show()
    #pause()
    if target == "dark" :
        img.save('recon_d/'+str(i)+'.jpg', 'JPEG')
    elif target == "noise" :
        img.save('recon_n/'+str(i)+'.jpg', 'JPEG')
    elif target == "combine" :
        img.save('recon_c/'+str(i)+'.jpg', 'JPEG')