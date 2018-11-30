from xml.etree import ElementTree
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import gensim.downloader as api
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Conv2D
from keras.models import Model
from keras.initializers import Constant

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

# 设置session
KTF.set_session(sess)

def readxml(name):
    tree = ElementTree.parse(name)
    dict = {}
    for elem in tree.iter(tag='table'):
        for table in elem.iter(tag='column'):
            if table.attrib['name'] not in dict:
                dict[table.attrib['name']] = []
                dict[table.attrib['name']].append(table.text)
            else:
                dict[table.attrib['name']].append(table.text)
    df = pd.DataFrame(dict)
    df = df.iloc[:, :].replace({'NULL': 0})
    if 'bank' in name:
        f = lambda r: int(r.sberbank) + int(r.vtb) + int(r.gazprom) + int(r.alfabank) + int(r.bankmoskvy) + int(
            r.raiffeisen) + int(r.uralsib) + int(r.rshb)
    else:
        f = lambda r: int(r.beeline) + int(r.mts) + int(r.megafon) + int(r.tele2) + int(r.rostelecom) + int(
            r.komstar) + int(r.skylink)
    df['labels'] = df.iloc[:, :].apply(f, axis=1)
    return df

df = readxml('tkk_train_2016.xml')
df_2 = readxml('bank_train_2016.xml')
df_3 = readxml('banks_test_etalon.xml')
dft = readxml('tkk_test_2016.xml')
dftt = readxml('banks_test_2016.xml')
dftt_2 = readxml('tkk_test_etalon.xml')
col = ['text', 'labels']
df_train = pd.concat([df[col], df_2[col]], ignore_index=True)
df_test = pd.concat([dft[col], dftt[col]], ignore_index=True)
df_test_eta = pd.concat([df_3[col], dftt_2[col]], ignore_index=True)
df_test = pd.concat([df_test[col], df_test_eta[col]], ignore_index=True)
print(df_train.shape, df_test.shape)


# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2')
# train_x = tfidf.fit_transform(df_train.text).toarray()
# test_x = tfidf.fit_transform(df_train.text).toarray()
# train_y = df_train.labels
# test_y = df_test.labels
# print(train_x.shape)
#
# clf = LogisticRegression(C=0.1).fit(X=train_x, y=train_y)
# pred = clf.predict(test_x)
# n = 0
# for i, ii in zip(pred, test_y):
#     if i == ii:
#         n += 1
# print(n/len(pred) * 100)
texts = np.hstack((df_train['text'].values, df_test['text'].values))
#texts = np.array([t.split(' ') for t in texts])
labels = np.hstack((df_train['labels'].values, df_test['labels'].values))
labels[labels < -1] = -1
labels[labels > 1] = 1
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 20
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(labels, num_classes=3)
print('Shape of data tensor:', data.shape, data.shape[0])
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
data = data[indices]
labels = labels[indices]
n_test = len(df_test)

x_train = data[:-n_test:, :]
y_train = labels[:-n_test, :]
x_val = data[-n_test:, :]
y_val = labels[-n_test:, :]
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
model = api.load("word2vec-ruscorpora-300")
word_vectors = model.wv
embeddings_index = {}
for word, vocab_obj in model.wv.vocab.items():
    if int(vocab_obj.index) < MAX_NB_WORDS:
        embeddings_index[word] = word_vectors[word]
del model, word_vectors
print('Found %s word vectors.' % len(embeddings_index))

num_words = min(MAX_NB_WORDS, len(word_index))  # 对比词向量字典中包含词的个数与文本数据所有词的个数，取小
embedding_matrix = np.zeros((num_words + 1, 300))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



def embeddings(embedding_layer, x_train, y_train, x_val, y_val, explain):
    print('Training model.')
    print('*'*40, explain, '*'*40)
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(3, 3, activation='relu')(embedded_sequences)
    x = GlobalMaxPooling1D()(x)


    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    print(model.summary())
    import keras.callbacks
    class TestCallback(keras.callbacks.Callback):
        def __init__(self, test_data):
            self.test_data = test_data

        def on_epoch_end(self, epoch, logs={}):
            x, y = self.test_data
            loss, acc = self.model.evaluate(x, y, verbose=0)
            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=1024,
              epochs=2,
              verbose=1,
              callbacks=[TestCallback((x_val, y_val))])
# embedding_layer = Embedding(num_words + 1, 300,
#                             weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
# embeddings(embedding_layer, x_train, y_train, x_val, y_val, explain='Pre-trained embedding')
# embedding_layer = Embedding(len(word_index) + 1, 300,
#                             embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None,
#                             embeddings_constraint=None, mask_zero=False, input_length=None)
# embeddings(embedding_layer, x_train, y_train, x_val, y_val, explain='Self-trained embedding')
'''
Epoch 1/2

 1024/18035 [>.............................] - ETA: 29s - loss: 1.0986 - acc: 0.6660
18035/18035 [==============================] - 2s 103us/step - loss: 1.0894 - acc: 0.6580

Testing loss: 1.0667365112442784, acc: 0.9493964613268617

Epoch 2/2

 1024/18035 [>.............................] - ETA: 0s - loss: 1.0811 - acc: 0.6719
18035/18035 [==============================] - 0s 3us/step - loss: 1.0766 - acc: 0.6580

Testing loss: 1.0446980882859693, acc: 0.9493964613268617
'''
'''

'''

MAX_WORD_LEN = 16
MAX_SENT_LEN = 21

# russian letters and basic punctuation
index2let = [chr(i) for i in range(1072, 1104)] + list(".,!?")
let2index = {let: i + 1 for i, let in enumerate(index2let)}
index2let = set(index2let)

LETTERS_COUNT = len(index2let)

def preprocess_word(word):
    word = [let2index[ch] for ch in word if ch in index2let]
    if len(word) <= MAX_WORD_LEN:
        word += [0 for _ in range(MAX_WORD_LEN - len(word))]
        return word
    return None

def preprocess(sent):
    sent = sent.lower()
    sent = ''.join([i if i in index2let else ' ' for i in sent])
    for i in ".,!?":
        sent = sent.replace(i, " {} ".format(i))
    sent = sent.split()
    sent = [preprocess_word(word) for word in sent]
    sent = [s for s in sent if s]
    sent += [[0 for _ in range(MAX_WORD_LEN)]
             for _ in range(MAX_SENT_LEN - len(sent))]
    return np.array(sent)


data = np.array([preprocess(line).reshape(-1) for line in texts])
new_data = np.zeros((len(data), len(data[0])))
for i, j in enumerate(data):
    new_data[i] = j
print(new_data.shape)
print(asdf)
indices = np.arange(data.shape[0])
data = data[indices]
labels = labels[indices]
x_train = data[:-n_test]
y_train = labels[:-n_test]
x_val = data[-n_test:]
y_val = labels[-n_test:]

print(x_train[0].shape)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

embedding_layer = Embedding(len(word_index) + 1, 300,
                            embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None,
                            embeddings_constraint=None, mask_zero=False, input_length=None)
embeddings(embedding_layer, x_train, y_train, x_val, y_val, explain='Character embedding')
