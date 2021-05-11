



# import os


# from data_helper import DataHelper
# from transformers import AutoTokenizer
#
# import keras
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# from sklearn.metrics import classification_report
#
# import tensorflow as tf
# import numpy as np
#
# import bert
#
# model_name = "albert_tiny"
# model_dir = bert.fetch_brightmart_albert_model(model_name, ".models")
# model_ckpt = os.path.join(model_dir, "albert_model.ckpt")
# bert_params = bert.params_from_pretrained_ckpt(model_dir)
# l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
#
# fully_model = keras.models.Sequential([
#     keras.layers.InputLayer(input_shape=(50,)),
#     l_bert,
#     keras.layers.Lambda(lambda x: x[:, 0, :]),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
# fully_model.build(input_shape=(None, 50))
# bert.load_albert_weights(l_bert, model_ckpt)
#
# dataHelper = DataHelper()
# contents, labels = dataHelper.get_data('test1.csv')
#
# max_length = 100
# tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
# tokens = tokenizer.batch_encode_plus(
#     contents,
#     max_length=max_length,
#     padding='max_length',
#     truncation=True,
#     return_tensors='tf'
# )
#
# input_ids = tokens['input_ids']
# tags = labels
# lb = preprocessing.LabelBinarizer()
# tags = lb.fit_transform(tags)
# print(lb.classes_)
# labels = []
# for index, label in enumerate(tags):
#     labels.append(label)
#
# labels = np.array(labels)
# input_ids = np.array(input_ids)
# x_train, x_test, y_train, y_test = train_test_split(input_ids, labels, test_size=0.2, random_state=1, shuffle=True)
# adam = keras.optimizers.Adam()
#
# # Fully :
# fully_model.compile(loss=keras.losses.binary_crossentropy, optimizer=adam, metrics=['accuracy'])
# # fully_model.summary()
#
# fully_model.fit(x=x_train, y=y_train, epochs=5, batch_size=32, verbose=1)
# fully_model.save('Fully_Bert')
# model = keras.models.load_model('Fully_Bert')
# pre = model.predict(x_test)
# predict = []
# for p in pre:
#     if p[0] >= 0.5:
#         predict.append([1])
#     elif p[0] < 0.5:
#         predict.append([0])
# predict = np.array(predict)
# fully_connected_report = classification_report(y_test, predict)
# print('fully_connected report : ')
# print(fully_connected_report)
#
#
# # RNN :
# RNN_model = keras.models.Sequential([
#     keras.layers.InputLayer(input_shape=(50,)),
#     l_bert,
#     keras.layers.Lambda(lambda x: tf.reshape(x[:, 0, :],[-1,1,312])), # [-1, 1, 312]
#     keras.layers.LSTM(128, input_shape=(1, 312), activation='relu', return_sequences=True),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
# RNN_model.build(input_shape=(None, 50))
#
# RNN_model.compile(loss=keras.losses.binary_crossentropy, optimizer=adam, metrics=['accuracy'])
# RNN_model.fit(x=x_train, y=y_train, epochs=3, batch_size=32, verbose=1)
#
# RNN_model.save('RNN_Bert')
# model = keras.models.load_model('RNN_Bert')
# pre = model.predict(x_test)
# predict = []
# for p in pre:
#     if p[0][0] >= 0.5:
#         predict.append([1])
#     elif p[0][0] < 0.5:
#         predict.append([0])
# predict = np.array(predict)
# RNN_report = classification_report(y_test, predict)
# print('RNN report : ')
# print(RNN_report)
