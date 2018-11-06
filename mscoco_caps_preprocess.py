import json
import pickle

with open('MSCOCO2014_data/annotations/captions_train2014.json') as f:
    train_captions = json.load(f)
with open('MSCOCO2014_data/annotations/captions_val2014.json') as f:
    val_captions = json.load(f)
val_captions = val_captions['annotations']
train_captions = train_captions['annotations']

id2cap_train = {}
id2cap_val = {}

for bcap in val_captions:
    img_id = bcap['image_id']
    cap = bcap['caption'].strip(". ")
    if img_id in id2cap_val:
        id2cap_val[img_id].append(cap)
    else:
        id2cap_val[img_id] = [cap]

for bcap in train_captions:
    img_id = bcap['image_id']
    cap = bcap['caption'].strip(". ")
    if img_id in id2cap_train:
        id2cap_train[img_id].append(cap)
    else:
        id2cap_train[img_id] = [cap]

pickle.dump( id2cap_train, open( "id2cap_train.p", "wb" ))
pickle.dump( id2cap_val, open( "id2cap_val.p", "wb" ))
