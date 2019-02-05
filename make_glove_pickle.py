import os
import pickle
import numpy as np

'''
This script converts the specified Glove file into a pickled dict
'''

embeddings_index = {}

with open('hi.vec', encoding="utf8") as glove_file:
    for line in glove_file:
        values = line.split()
        word = values[0]
        #print(values[1:])
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

if not os.path.exists('PickledData/'):
    print('MAKING DIRECTORY PickledData/ to save pickled glove file')
    os.makedirs('PickledData/')

with open('PickledData/Glove.pkl', 'wb') as f:
    pickle.dump(embeddings_index, f)

print('SUCESSFULLY SAVED Glove data as a pickle file in PickledData/')
