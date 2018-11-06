import json
#from pprint import pprint
import pickle

with open('MSCOCO2014_data/annotations/captions_train2014.json') as f:
    train_captions = json.load(f)
with open('MSCOCO2014_data/annotations/captions_val2014.json') as f:
    val_captions = json.load(f)
val_captions = val_captions['annotations']
train_captions = train_captions['annotations']

big_list_words = set([])

for bcap in val_captions:
    cap = bcap['caption'].strip(". ")
    big_list_words.update(cap.split())

for bcap in train_captions:
    cap = bcap['caption'].strip(". ")
    big_list_words.update(cap.split())

big_list_words = list(big_list_words)
big_list_words.insert(0,'<end>')
big_list_words.insert(0,'<start>')

vocab = {}

word2id = { wd : _id for _id, wd in enumerate(big_list_words) }

vocab['id2word'] = big_list_words
vocab['word2id'] = word2id

pickle.dump( vocab, open( "vocab.p", "wb" ) )




