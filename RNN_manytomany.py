import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import SimpleRNN,TimeDistributed,Embedding, Dense
from pprint import pprint


def plot_history(histories, key='loss'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history[ key],
                       '--', label=name.title() )
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()

def preprocess():
    sentences = [['I', 'feel', 'hungry'],
                 ['tensorflow', 'is', 'very', 'difficult'],
                 ['tensorflow', 'is', 'a', 'framework', 'for', 'deep', 'learning'],
                 ['tensorflow', 'is', 'very', 'fast', 'changing']]
    pos = [['pronoun', 'verb', 'adjective'],
           ['noun', 'verb', 'adverb', 'adjective'],
           ['noun', 'verb', 'determiner', 'noun' ,'preposition', 'adjective', 'noun'],
           ['noun', 'verb', 'adverb', 'adjective', 'verb']]

    #word to index matrix를 만들기
    word_list = sum(sentences, [])
    word_list = sorted(set(word_list))
    word_list = ['<pad>'] + word_list
    word2idx = {word : idx for idx, word in enumerate(word_list)}
    idx2word = {idx : word for idx, word in enumerate(word_list)}


    #pos(part-of-speech) to index matrix를 y-data만들기
    pos_list = sum(pos, [])
    pos_list = sorted(set(pos_list))
    pos_list = ['<pad>'] + pos_list
    pos2idx = {pos : idx for idx, pos in enumerate(pos_list)}
    idx2pos = {idx : pos for idx, pos in enumerate(pos_list)}

    global input_dim, output_dim, max_sequence, hidden_dim, one_hot, num_classes
    # converting sequence of tokens to sequence of indices
    max_sequence = 10
    x_data = list(map(lambda sentence : [word2idx.get(token) for token in sentence], sentences))
    y_data = list(map(lambda sentence : [pos2idx.get(token) for token in sentence], pos))
    # padding the sequence of indices
    x_data = pad_sequences(sequences=x_data, maxlen=max_sequence, padding='post')
    x_data_mask = ((x_data != 0) * 1).astype(np.float32)
    x_data_len = list(map(lambda sentence : len(sentence), sentences))
    y_data = pad_sequences(sequences=y_data, maxlen=max_sequence, padding='post')

    y_data= to_categorical(y_data)
    # creating rnn for "manny to many" sequence tagging

    num_classes = len(pos2idx)
    hidden_dim = 10
    input_dim = len(word2idx)
    output_dim = len(word2idx)
    one_hot = np.eye(len(word2idx))

    return x_data, y_data

def make_model(x_data, y_data):
    model = Sequential()

    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, mask_zero=True,
                               trainable=False, input_length=max_sequence,
                               embeddings_initializer=keras.initializers.Constant(one_hot)))
    model.add(SimpleRNN(units=hidden_dim, return_sequences=True))
    model.add(TimeDistributed(Dense(units=num_classes,activation='softmax')))
    model.summary()
    model.compile(loss      =  'categorical_crossentropy',
                      optimizer =  'adam',
                      metrics   =  ['acc'])
    rnn_training = model.fit(x_data, y_data, batch_size=128, epochs=1000)
    plot_history([('Normal', rnn_training)])


def main():
    x_data, y_data= preprocess()
    make_model(x_data, y_data)

main()