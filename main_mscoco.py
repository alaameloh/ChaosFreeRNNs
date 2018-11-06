import json
import os
import tensorflow as tf
import pickle
import numpy as np
import pickle
from random import shuffle
from CFN_impl import CFNCell
import math

## debug
from random import randint

def get_id(imname):
    """ get id from img's name of MSCOCO. 
    """
    imname = imname.replace(".jpg",'')
    return int(imname.split('_')[2])

########################################################################################
## Parameters to change
########################################################################################
DEBUG = True

# this var is to set up the id for model saved by tensorflow
start_index = 0

# how many number of epochs we will train start from start_index.
nb_epochs = 0

## state size of RNNCell
hidden_sz = 256

## where to store extracted cnn features
filename = "//datas/teaching/projects/spring2018/ps54/MSCOCO_trained_CNN_features/feats_mscoco_49c_ft.json"

## maximum length of captions, for mscoco, it is 50
max_length = 55

## size of CNN feature is extracted.
fts_size = 2048 ## assume fts have rank = 1

fixed_batch_size = 60

# these two parameter is the probabilty of dropouts in CFN.
p = 0.55
q = 0.45

########################################################################################

## load CNN fts into mscoco_vecs
## Form of mscoco_vecs = {id : cnn_vecs} 
mscoco_vecs = dict()
with open(filename) as f:
    c = 0
    for line in f:
        imname, vec = json.loads(line.strip())
        id = get_id(imname)
        mscoco_vecs[id] = np.array(vec)

# load mapper of id to captions
id2cap_train = pickle.load( open( "id2cap_train.p", "rb" ))
id2cap_val = pickle.load(open("id2cap_val.p","rb"))
vocab = pickle.load( open( "vocab.p", "rb" ))

id_start = vocab['word2id']['<start>']
id_end = vocab['word2id']['<end>']

class Dataset:
    """ Mimic the dataset class of Tensorflow to generate the batch.
    """
    def __init__(self, _cnn_vecs, _id2caps, _vocab, batch_size, debug = DEBUG):
        self.cnn_vecs = _cnn_vecs
        self.id2cap = _id2caps
        self.vocab = _vocab 
        self.ids = list(_id2caps.keys())
        self.cur_id = 0
        self.cur_id_cap = 0
        self.bsz = batch_size
        self.nb_caps_last_image = len(self.id2cap[self.ids[-1]])

    def reset(self):
        self.cur_id_cap = 0
        self.cur_id = 0
        shuffle(self.ids)

    def next_batch(self):
        """ Generate next batch for training set.
        """
        fts = []
        caps = []
        stn_length = []
        nb_added = 0

        while nb_added < self.bsz and self.cur_id < len(self.ids):
            if  self.cur_id == len(self.ids) - 1 and self.cur_id_cap == self.nb_caps_last_image:
                break
            cur_id_img = self.ids[self.cur_id]
            if cur_id_img not in self.id2cap:
                self.cur_id_cap = 0
                self.cur_id += 1
                continue
            max_cap = len(self.id2cap[cur_id_img])
            if self.cur_id_cap == max_cap:
                self.cur_id_cap = 0
                self.cur_id+=1
                continue
            fts.append(self.cnn_vecs[cur_id_img])
            cap = self.id2cap[cur_id_img][self.cur_id_cap]
            cap = "<start> " + cap + " <end>"
            embs = [self.vocab['word2id'][w] for w in cap.split()]
            stn_length.append(len(embs))
            caps.append(embs)
            nb_added+=1
            self.cur_id_cap+=1
        if nb_added !=  self.bsz:
            self.reset()
            return None, None, None
        return fts, caps, stn_length
    
    def next_batch_val(self):
        """ next validation batch, the ouput return is different with next_batch()
        """
        if self.cur_id + self.bsz >= len(self.id2cap):
            return None, None
        start = self.cur_id
        end = self.cur_id + self.bsz
        self.cur_id = end 
        ids = self.ids[start: end]
        return [self.cnn_vecs[id_] for id_ in ids], ids
    
    def random_batch(self):
        """for debug only"""
        fts = []
        caps = []
        stn_length = []
        for _ in range(self.bsz):
            x = randint(0, max_length - 2)
            caps.append(np.random.randint(0, 10000,size=x))
            stn_length.append(x)
            fts.append(np.random.rand(fts_size,))
        return fts, caps, stn_length

# load dataset for training
dataset = Dataset(mscoco_vecs,id2cap_train,vocab,fixed_batch_size)

# load dataset for evaluating.
val_dataset = Dataset(mscoco_vecs,id2cap_val,vocab,fixed_batch_size)            


def data_type():
    return tf.float32

## masking, padding all in one.
def padding_masking(fts, caps, cap_length):
    padded = np.array([ cap + [0]* (max_length - len(cap)) for cap in caps])
    masked = np.array([ [1.0] * l + [0.0] * (max_length - l) for l in cap_length])
    arr_fts = np.array(fts)
    return arr_fts, padded, masked

vocab_sz = len(vocab['word2id'])

cnn_fts = tf.placeholder(dtype = data_type(), shape=[fixed_batch_size, fts_size])
out_caps = tf.placeholder(dtype = tf.int32, shape=[fixed_batch_size, max_length])
mask = tf.placeholder(dtype = data_type(), shape=[fixed_batch_size, max_length])
mode = tf.placeholder(tf.int32, shape=[], name="condition")
    
class CFNModel:
    def __init__(self,p,q):
        self.weights = {
                            ## Maybe wrong at init value
                            'W0': tf.get_variable('W0',[vocab_sz, hidden_sz], initializer=tf.random_uniform_initializer(dtype = data_type(), minval=-0.07,maxval=0.07)), ## None for batch size | embed_sz
                            #'W1': tf.get_variable('W1',[cnn_ft_size, hidden_sz], initializer=tf.random_uniform_initializer(dtype = data_type(), minval=-0.07,maxval=0.07)),
                            'W3': tf.get_variable('W3',[hidden_sz, vocab_sz], initializer=tf.random_uniform_initializer(dtype = data_type(), minval=-0.07,maxval=0.07))
                        }
        self.biases = {
            'b3': tf.get_variable('b3',[vocab_sz], initializer=tf.random_uniform_initializer(dtype = data_type(), minval=-0.07,maxval=0.07))
        }
        self.cell = CFNCell(hidden_sz,p,q)
        self.cell.build([1,1])
        self.reuse = False

    def init_state(self,cnn_fts_):
        dense = tf.layers.dense(inputs=cnn_fts_, units=hidden_sz, activation=tf.nn.relu, name = 'cnn_rnn_map', reuse=self.reuse)
        zero_state_ = self.cell.zero_state(fixed_batch_size, data_type())
        init_state, _ = self.cell.call(dense,zero_state_)
        self.reuse = True
        return init_state

    def train(self,cnn_fts_,target_, mask):
        # convert inputs to one-hot. 
        inputs = tf.one_hot(target_[:,:-1], depth = vocab_sz)
        # just for mutiply between 2D and 3D
        mask3d = tf.expand_dims(mask, 2)
        inputs = tf.multiply(inputs, mask3d[:,1:,:])
        # embedding steps.
        middle_inputs = tf.einsum('ijk,kl->ijl', inputs, self.weights['W0'],name = 'embbeding')
        ## break down a batch of words into list of words (size= time_step)
        middle_inputs = tf.unstack(middle_inputs, num=max_length - 1, axis=1) # middle_inputs
        i = 1
        outputs = []
        state = self.init_state(cnn_fts_)

        for inp in middle_inputs:
            cur_state, _ = self.cell(inp,state)
            t = tf.expand_dims(mask[:,i],1)
            state = cur_state * t + (1.0 - t)* state
            outputs.append(state)

        output = tf.stack(outputs, axis = 1) #[batch_size, T, hidden_state] * [batch_szie] = [batch_size, T, hidden_state]
        
        mm = tf.einsum('ijk,kl->ijl', output, self.weights['W3'],name="logits_weights")
        logits = mm + self.biases['b3']
        return tf.contrib.seq2seq.sequence_loss(
                                                    logits,
                                                    target_[:,1:],
                                                    mask[:,1:], # tf.ones([fixed_batch_size, max_length - 1], dtype=data_type())
                                                    average_across_timesteps=True,
                                                    average_across_batch=True
                                                )

    def predict(self, cnn_fts_):
        # generate the start word for every element in batch
        inp = tf.one_hot([id_start]*fixed_batch_size,depth = vocab_sz)
        inp = tf.reshape(inp, [fixed_batch_size,vocab_sz])
        state = self.init_state(cnn_fts_)

        predictions = []
        for i in range(1,max_length):
            inp = tf.einsum('ik,kl->il', inp, self.weights['W0'])
            state, _ = self.cell(inp,state)
            mm = tf.einsum('ik,kl->il', state, self.weights['W3'])
            logit = mm + self.biases['b3']
            prediction = tf.argmax(logit,axis = 1)
            predictions.append(prediction)
            inp = tf.one_hot(prediction,vocab_sz)
        predictions = tf.stack(predictions, axis = 1)
        return predictions

# place holder for learning rate so we could change it to each epochs.
place_holder_lr = tf.placeholder(tf.float32, shape=[])

# set up the model.
with tf.device('/gpu:0'):
    cfn = CFNModel(p,q)
    loss = cfn.train(cnn_fts, out_caps,mask)
    predictions = cfn.predict(cnn_fts)
    optimizer = tf.train.AdagradOptimizer(place_holder_lr)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    train = optimizer.apply_gradients(zip(gradients, variables))

logs_path = './log_files/'

init = tf.global_variables_initializer()
# Create a summary to monitor cost tensor
tf.summary.scalar("Loss", loss)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

# initiated learning rate.
lr = 0.007
# the reduce factor of learning rate if valid perplex didnt reduced at least 1%
reduce_lr_factor = 1.1

sess = tf.Session()
sess.run(init)

model_path = "./models/ImageCaptioning/ImgCap_CFN.ckpt" 

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
  print("Start epoch: ", eps)
  i = 0
  while True:
    cnn, caps, msk = dataset.next_batch() # if not DEBUG else dataset.random_batch() #
    if not (cnn and caps and msk):
        break
    arr_cnn, arr_caps, arr_msk = padding_masking(cnn, caps, msk)
    _, train_loss, summary = sess.run([train, loss, merged_summary_op], feed_dict= {cnn_fts : arr_cnn, out_caps : arr_caps, mask: arr_msk, place_holder_lr : lr, mode : 1})

    # Write logs at every iteration
    summary_writer.add_summary(summary, i)
    if (i%print_step ==0):
        print("Loss : ", train_loss)#/weight)
    i+=1
  save_path = saver.save(sess, model_path ,global_step=eps)
  print("Model saved in path: %s" % save_path)
  print("Finished epoch: ", eps)
  print("==============================================================================")  

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>") 

# start evaluation loss.
prediction_caps = {}
ide = 0
while True:
    fts, ids = val_dataset.next_batch_val()
    if not fts:
        break
    arr_cnn = np.array(fts) 
    preds = sess.run(predictions, feed_dict= {cnn_fts : arr_cnn})
    temp = preds.tolist()
    #prediction_caps[:] = []
    for id_, cap in zip(ids, temp): 
        prediction_caps[id_] = cap
    print("Finish ",ide, " batches")
    ide+=1
        
pickle.dump( prediction_caps, open( "mscoco_predictions.p", "wb" ))
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("Finished!")
