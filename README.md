# POS-Tagging-for-Hindi
Developing a POS tagger for Hindi Copora

# Implementations
  - [x] Bi-LSTM Architecture
  - [x] Bi-LSTM + CRF Architecture
  - [ ] Residual LSTM + EMLO (Testing)
  
# TL;DR
  
1) `extract_tags.sh` and `extract_data` are used to extract data from the Hindi Corpora
2) Currently [Hindi word embeddings](https://www.dropbox.com/s/pq50ca4o3phi9ks/hi.tar.gz?dl=0) trained on Fasttext are used. 
3) `train.py` files contains the implementation of the above architectures.
# Requirements
Requirements:

* Python 3.6 
* Keras 2.2.0 - For the creation of BiLSTM-CRF architecture
* Tensorflow 1.8.0 - As backend for Keras (other backends are untested.

# Resuts
![picture alt](https://github.com/rootally/POS-Tagging-for-Hindi/blob/master/Bi-Lstm%2BCRF.png "Bi-LSTM + CRF")
TEST ACCURACY with Bi-LSTM + CRF : 0.977324
