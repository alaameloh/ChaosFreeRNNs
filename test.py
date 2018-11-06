#!/usr/bin/env python3
"""
Author: Hung NGUYEN
EURECOM - June 2018

This script contains implemetation of CFN network which is mentioned in the paper: 
"A recurrent neural network without chaos" - Thomas Laurent, James von Brecht

Link: https://arxiv.org/abs/1612.06212
Only use for research.

"""

# these lib comes from official python
import math
import time
import os

# common machine learning libs
import tensorflow as tf
import numpy as np

# implememtation of CFNCell
# in the same folder with this script.
from CFN_impl import CFNCell

# this script is only for loading PTB dataset.
# It is the same folder with this script. 
import reader


## setup all the parameter for this script
DATA_PATH = 'data/'
batch_sz = 20
time_step = 35
hidden_sz = 731

## This parameter will not use.
# embed_sz = 10 ## 10000**0.25


# these two parameters belows is for manual guiding training.
# Training process is composed of mutiple sessions.
# These two parameters must be set manually.

# the number of epochs of the current session
nb_epochs = 5
# The last index of saved checkpoint will be start_index - 1
# so this session we start with start_index
start_index = 0

# return type of data in this network.
def data_type():
    return tf.float32

## load the dataset
raw_data = reader.ptb_raw_data(DATA_PATH)
train_data, valid_data, test_data, vocab_sz = raw_data


class Formatter:
    """ This class's purpose is to use function convert_to_np_array to reshape
        given data to match with input format of CFN network.
    """
    def __init__(self, time_step):
        self.time_step = time_step

    def convert_to_np_array(self, _input):
        """ Reshape list to form the array shape (None, time_step) - shape's convention of numpy. 
        """
        nb_row = (len(_input)-1)// time_step
        limit = nb_row * time_step
        return np.reshape(_input[:limit],(-1,time_step)), np.reshape(_input[1:limit+1],(-1,time_step))

## reformat all the data to numpy array
fmt = Formatter(time_step)
train_data_array, train_label_array = fmt.convert_to_np_array(train_data)
valid_data_array, valid_label_array = fmt.convert_to_np_array(valid_data)
test_data_array, test_label_array = fmt.convert_to_np_array(test_data)

## Place holder for input to train network.
features_placeholder = tf.placeholder(train_data_array.dtype, (None, time_step))
labels_placeholder = tf.placeholder(train_label_array.dtype, (None, time_step))

## other way to feed input to network. But it is not prefered.
#iterator = tf.data.Iterator.from_structure(features_placeholder.output_types,
                                        #    features_placeholder.output_shapes)

## get the data from placeholders, feed it to dataset. Batched data is fed
## to CFN network. next_elememt is fed to CFN network.
training_dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
training_dataset = training_dataset.batch(batch_sz)
iterator = training_dataset.make_initializable_iterator()
next_element = iterator.get_next()
training_init_op = iterator.make_initializer(training_dataset)

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Other way to feed data to CFN.
## Keep only for references.

# raw_data = tf.placeholder(tf.float32, shape=[None,]) # train_data
# #raw_data = tf.convert_to_tensor(raw_input_data)
# data_len = tf.placeholder(tf.int32) #tf.size(raw_data) # len(raw_data)
# #print(data_len)
# iter_len = data_len// time_step
# limit = iter_len//batch_sz* batch_sz

# # genertor to iterate the data
# def generator(): ## raw_data, time_step
#     for i in range(iter_len.value):
#         if i < limit:
#             yield (raw_data[i*time_step:(i+1)*time_step], raw_data[i*time_step+1:(i+1)*time_step + 1])
    
# #def _parse(f,l):
# #  return f, l

# ds = tf.data.Dataset.from_generator(
#     generator, (tf.int32, tf.int32), (tf.TensorShape([None]), tf.TensorShape([None]))) ## train_data, 20
# iterator = ds.batch(batch_sz).make_one_shot_iterator()
# value = iterator.get_next() ## .map(_parse)

## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## Mainly implementation of model.
def model(value,p,q):
    """ CFN model with the implementation of CFN cell
    """
    input_, target_ = value
    #assert tf.shape(input_)[0] == batch_sz, "size at current batch"

    ## embedding layer for input which is a word.
    ## shouldn'tbe needed for CNN cases.type
    # with tf.device("/cpu:0"):
    #     embedding = tf.get_variable(
    #        "embedding", [vocab_sz, hidden_sz], dtype=data_type())
    #     inputs = tf.nn.embedding_lookup(embedding, input_)
        
    ## be lazy to re-use this name
    inputs = tf.one_hot(input_, depth = vocab_sz)
    
    # Define weights
    # W0 is no longer used.
    weights = {
        ## Maybe wrong at init value
        'W0': tf.get_variable('W0',[vocab_sz, hidden_sz], initializer=tf.random_uniform_initializer(dtype = data_type(), minval=-0.07,maxval=0.07)), ## None for batch size | embed_sz
        'W3': tf.get_variable('W3',[hidden_sz, vocab_sz], initializer=tf.random_uniform_initializer(dtype = data_type(), minval=-0.07,maxval=0.07))
    }
    biases = {
        'b3': tf.get_variable('b3',[vocab_sz], initializer=tf.random_uniform_initializer(dtype = data_type(), minval=-0.07,maxval=0.07))
    }

    ## combine two cells together
    cell = tf.contrib.rnn.MultiRNNCell([CFNCell(hidden_sz,p,q) for _ in range(2)], state_is_tuple=True)
    ## using einsum to do matrix multiply between 3D Tensor and 2D Tensors
    middle_inputs = tf.einsum('ijk,kl->ijl', inputs, weights['W0'])
    
    ## init the value for state( all element is zeros)
    _initial_state = cell.zero_state(batch_sz, data_type())
    
    ## break down a batch of words into list of words (size= time_step)
    middle_inputs = tf.unstack(middle_inputs, num=time_step, axis=1) # middle_inputs

    ## get output and state after 2 CFN cells
    outputs, state = tf.nn.static_rnn(cell, middle_inputs,
                                      initial_state=_initial_state)
    
    ## outputs and states are in form a "list" of Tensor 

    ## DEBUG purpose
    #output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_sz])
    ##print(type(outputs[0]))
    
    # stack back the output to make batch of output
    output = tf.stack(outputs, axis = 1)

    # apply the last layers
    mm = tf.einsum('ijk,kl->ijl', output, weights['W3'])

    ## apply the bias terms
    logits = mm + biases['b3']   ## tf.matmul(weights['W3'], outputs) +


    logits = tf.reshape(logits, [batch_sz, time_step, vocab_sz])
    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        target_,
        tf.ones([batch_sz, time_step], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    _cost = tf.reduce_sum(loss)
    _final_state = state
    
    return _cost, _final_state, logits

## two dropout probs for two gates.
p = 0.55
q = 0.45

# place holder for learning rate so we could change it to each epochs.
place_holder_lr = tf.placeholder(tf.float32, shape=[])

# set up the model.
with tf.device('/gpu:0'):
    loss, _,logits = model(next_element,p,q)

## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# # set up for training.
# optimizer = tf.train.GradientDescentOptimizer(place_holder_lr) #   AdagradOptimizer
# gvs = optimizer.compute_gradients(loss)
# # this line below is used for the case of embedding.
# capped_gvs = [(tf.clip_by_norm(grad, 1.), var) if grad is not None else (grad, var) for grad, var in gvs]
# train = optimizer.apply_gradients(capped_gvs)

## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## set up for training.
optimizer = tf.train.AdagradOptimizer(place_holder_lr)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
train = optimizer.apply_gradients(zip(gradients, variables))


logs_path = './log_files/'

init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.summary.scalar("Loss", loss)
# Create a summary to monitor accuracy tensor
#tf.summary.scalar("Accuracy", acc)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

# initiated learning rate.
lr = 0.7
# the reduce factor of learning rate if valid perplex didnt reduced at least 1%
reduce_lr_factor = 1.1

sess = tf.Session()
sess.run(init)

model_path = "./models/global_n_dense/CFN.ckpt" 

# check if check-point file exists. If yes, load it.
ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_path))
if ckpt and ckpt.model_checkpoint_path:
  saver.restore(sess, ckpt.model_checkpoint_path)

# last eval perplexity
last_eval_perp = 0

# the number of epochs to display the results.
print_step = 100

#Compute for 100 epochs.
for eps in range(start_index, start_index + nb_epochs):
  print("==============================================================================")
  ## legacy.
  #sess.run(iterator.initializer)
  print("Start epoch: ", eps)
  sess.run(training_init_op, feed_dict={features_placeholder : train_data_array, labels_placeholder : train_label_array})
  i = 0
  while True:
    try:
      _, train_loss, summary = sess.run([train, loss, merged_summary_op], feed_dict= {features_placeholder : train_data_array, labels_placeholder : train_label_array, 
                                                                                   place_holder_lr : lr})
      #{raw_data :train_data, place_holder_lr : lr0, data_len: len(train_data)})#, feed_dict={raw_data : tf.convert_to_tensor(train_data)})

      # Write logs at every iteration
      summary_writer.add_summary(summary, i)
      if (i%print_step ==0):
        print("Loss : ", train_loss/time_step)
      i+=1
    except : # tf.errors.OutOfRangeError
      break
    
  print("Done training for epoch")

  # evaluate after each epochs to get what is the perplexity on validation set
  sess.run(training_init_op, feed_dict={features_placeholder : valid_data_array, labels_placeholder : valid_label_array})
  current_eval_perp = []
  # start evaluation loss.
  while True:
    try:
      eval_loss = sess.run(loss, feed_dict= {features_placeholder : valid_data_array, labels_placeholder : valid_label_array})
      current_eval_perp.append(eval_loss)
      
    except: #Exception as ex: # tf.errors.OutOfRangeError
      #print(ex)
      break
  cur_eval = math.exp(sum(current_eval_perp)/(len(current_eval_perp)* time_step))
  if eps > 0:
      if cur_eval > 0.99 * last_eval_perp:
          lr =  lr/reduce_lr_factor
  last_eval_perp = cur_eval
  print("Eval perplex: ",cur_eval)
  #timestr = time.strftime("%Y%m%d-%H%M%S")
  save_path = saver.save(sess, model_path ,global_step=eps)
  print("Model saved in path: %s" % save_path)
  print("Finished epoch: ", eps)
  print("==============================================================================")


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>") 
sess.run(training_init_op, feed_dict={features_placeholder : test_data_array, labels_placeholder : test_label_array})
test_perp = []
# start evaluation loss.
while True:
    try:
      test_loss = sess.run(loss, feed_dict= {features_placeholder : test_data_array, labels_placeholder : test_label_array})
      test_perp.append(test_loss)
    except:
      break
test_res = math.exp(sum(test_perp)/(len(test_perp)* time_step))
print("Test perplex: ",test_res)
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("Finished!")


    


    