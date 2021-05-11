import os
import datetime

import jieba as jieba
from transformers import AutoTokenizer

import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

import tensorflow as tf
import numpy as np

import bert


class AlbertModel:
    def __init__(self):
        model_name = "albert_tiny"
        model_dir = bert.fetch_brightmart_albert_model(model_name, ".models")
        self.model_ckpt = os.path.join(model_dir, "albert_model.ckpt")
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        self.l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        self.l_bert.trainable = True
        self.adam = keras.optimizers.Adam()

    def tokenizeData(self,contents,max_length):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        tokens = tokenizer.batch_encode_plus(
            contents,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        input_ids = tokens['input_ids']
        return np.array(input_ids)

    def LabelBinarizer(self,labels):
        lb = preprocessing.LabelBinarizer()
        tags = lb.fit_transform(labels)
        labels = []
        for index, label in enumerate(tags):
            labels.append(label)
        return np.array(labels)

    def fully(self,contents,labels):
        max_length = 100
        contents = self.tokenizeData(contents,max_length)
        labels = self.LabelBinarizer(labels)
        x_train, x_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, random_state=1,
                                                            shuffle=True)
        fully_model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(max_length,)),
            self.l_bert,
            keras.layers.Lambda(lambda x: x[:, 0, :]),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        fully_model.build(input_shape=(None, max_length))
        bert.load_albert_weights(self.l_bert, self.model_ckpt)


        # Tensorboard
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        fully_model.compile(loss=keras.losses.binary_crossentropy, optimizer=self.adam, metrics=['accuracy'])
        fully_model.fit(x=x_train, y=y_train,
                        epochs=3,
                        batch_size=32,
                        verbose=1,
                        validation_data=(x_test,y_test),
                        callbacks=[tensorboard_callback])

        score = fully_model.evaluate(x_test,y_test,verbose=0)
        results = dict(zip(fully_model.metrics_names, score))
        print(results)

        self.save(fully_model,'Fully_Albert')
        self.evaluate('Fully_Albert',x_test,y_test)

    def rnn(self,contents,labels):
        max_length = 100
        contents = np.array(self.tokenizeData(contents, max_length))
        labels = np.array(self.LabelBinarizer(labels))
        x_train, x_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, random_state=1,
                                                            shuffle=True)
        RNN_model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(max_length,)),
            self.l_bert,
            keras.layers.Lambda(lambda x: tf.reshape(x[:, 0, :], [-1, 1, 312])),  # [-1, 1, 312]
            keras.layers.LSTM(128, input_shape=(1, 312), activation='relu', return_sequences=True),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        RNN_model.build(input_shape=(None, max_length))
        bert.load_albert_weights(self.l_bert, self.model_ckpt)

        # Tensorboard
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


        RNN_model.compile(loss=keras.losses.binary_crossentropy, optimizer=self.adam, metrics=['accuracy'])
        RNN_model.fit(x=x_train, y=y_train,
                      epochs=3,
                      batch_size=32,
                      verbose=1,
                      validation_data=(x_test,y_test),
                      callbacks=[tensorboard_callback])

        self.save(RNN_model,'RNN_Albert')
        self.evaluate('RNN_Albert',x_test,y_test)

    def save(self,model,model_name):
        model.save(model_name)

    def load(self,model_name):
        self.model = keras.models.load_model(model_name)

    def predict(self,model_name,predict_data):
        # predict_data = self.tokenizeData(predict_data,100)
        self.load(model_name)
        pre = self.model(predict_data)
        return pre

    def evaluate(self,model_name,x_test,y_test):
        pre = self.predict(model_name,x_test)
        pre = np.array(pre)
        pre = pre.reshape(len(pre),1)
        predict = []
        for p in pre:
            predict.append(round(float(p[0])))
        report = classification_report(y_test,predict)
        print(report)