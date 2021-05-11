from tqdm import keras
from transformers import BertTokenizerFast, AutoModel,AdamW
import torch
from torch.utils.data import TensorDataset,DataLoader,RandomSampler, SequentialSampler

import tensorflow_hub as hub
from keras.models import Sequential
from keras.layers import Dense,Input,LSTM,Embedding,Dropout
import keras

from sklearn import svm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import joblib
import numpy as np
import torch.nn as nn

class ModelHelper:
    def feature_base_fit(self,contents,labels):
        features = np.array(self.tokenize_sentence(contents))
        labels = np.array(self.LabelBinarizer(labels))
        print(features.shape)
        print(labels.shape)
        x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=1,shuffle=True)

        print('fully_connected : ')
        self.fully_connected(x_train,y_train)
        model = keras.models.load_model('Fully_Bert')
        pre = model.predict(x_test)
        predict = []
        for p in pre:
            if p[0] >= 0.5:
                predict.append([1])
            elif p[0] < 0.5:
                predict.append([0])
        fully_connected_report = classification_report(y_test,predict)

        print('RNN : ')
        self.RNN(x_train,y_train)
        model = keras.models.load_model('RNN_Bert')
        x_test = x_test.reshape(-1,1,312)
        pre = model.predict(x_test)
        predict = []
        for p in pre:
            if p[0][0] >= 0.5:
                predict.append([1])
            elif p[0][0] < 0.5:
                predict.append([0])
        RNN_report = classification_report(y_test, predict)

        print('fully_connected report : ')
        print(fully_connected_report)
        print('RNN report : ')
        print(RNN_report)


        # svm_model = svm.SVC(kernel='linear',probability=True)
        # xg_model = XGBClassifier(n_estimators=100,learning_rate=0.1)
        # lg_model = LogisticRegression()
        #
        # models = svm_model
        #
        # models.fit(x_train,y_train)
        # prediction = models.predict(x_test)
        # prob = models.predict_proba(x_test)
        # report = classification_report(y_test,prediction,output_dict=True)
        # print(report)

    def tokenize_sentence(self,contents):
        pretrain_model_name = 'voidful/albert_chinese_tiny'
        tokenizer = BertTokenizerFast.from_pretrained(pretrain_model_name)
        model = AutoModel.from_pretrained(pretrain_model_name)
        # para = list(models.named_parameters())
        # for p in para[-4:]:
        #     print(p[0])
        #     print(str(tuple(p[1].size())))
        max_length = 50
        features = []
        for content in contents:
            bert_input = tokenizer.encode_plus(
                content,
                truncation=True,
                max_length = max_length,
                padding='max_length',
                return_attention_mask=True
            )

            # for key in bert_input.keys():
            #   print(f'{key} : {bert_input[key]}')

            tokens_tensor = torch.tensor([bert_input['input_ids']])
            segments_tensors = torch.tensor([bert_input['attention_mask']])

            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors).pooler_output    # 將每個word轉為768維的vector 312?
                # feature = outputs[:,0,:].numpy()[0]                                            # 抽出[cls]的vector
                features.append(outputs[0].numpy())
        return features

    def fully_connected(self,x_train,y_train):
        print("x_train shape : ",x_train.shape)
        print("y_train shape : ", y_train.shape)

        model = Sequential()
        model.add(Input(shape=(312,)))
        model.add(Dense(units=128,activation='relu'))
        model.add(Dense(units=1,activation='sigmoid'))

        adam = keras.optimizers.Adam()

        model.compile(loss=keras.losses.binary_crossentropy,optimizer=adam,metrics=['accuracy'])
        model.summary()

        model.fit(x = x_train,y = y_train,epochs=200,batch_size=32,verbose=1)
        model.save('Fully_Bert')

    def RNN(self,x_train,y_train):
        x_train = x_train.reshape(-1,1,312)     # batch size , timestep , shape

        model = Sequential()
        model.add(LSTM(128,input_shape = (1,312),activation='relu',return_sequences=True))

        model.add(Dense(32,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))

        optimizer = keras.optimizers.Adam()
        model.compile(loss = keras.losses.binary_crossentropy,optimizer=optimizer,metrics=['accuracy'])
        model.fit(x_train,y_train,epochs=200,batch_size=32,verbose=1)

        model.summary()
        model.save('RNN_Bert')


    def fine_tunning_fit(self,contents,labels):
        pretrain_model_name = 'voidful/albert_chinese_tiny'
        tokenizer = BertTokenizerFast.from_pretrained(pretrain_model_name)
        bert = AutoModel.from_pretrained(pretrain_model_name)

        labels = self.LabelBinarizer(labels)
        # for i in range(len(contents)):
        #     print(contents[i],labels[i])

        x_train, x_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, random_state=1,
                                                            shuffle=True)

        # text = ["this is a bert models tutorial", "we will fine-tune a bert models"]
        # sent_id = tokenizer.batch_encode_plus(text, padding=True)
        # print(sent_id)

        tokens_train = tokenizer.batch_encode_plus(     # dict : include input_ids,token_type_ids,attention_mask
            x_train,
            max_length=25,
            padding=True,
            truncation=True
        )
        tokens_val = tokenizer.batch_encode_plus(
            x_test,
            max_length=25,
            padding=True,
            truncation=True
        )

        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(y_train)

        val_seq = torch.tensor(tokens_val['input_ids'])
        val_mask = torch.tensor(tokens_val['attention_mask'])
        val_y = torch.tensor(y_test)

        batch_size = 32

        train_data = TensorDataset(train_seq, train_mask, train_y)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        val_data = TensorDataset(val_seq, val_mask, val_y)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

        model = BERT_Arch(bert)
        optimizer = AdamW(model.parameters(),lr=1e-5)
        epochs = 10
        cross_entropy = nn.NLLLoss()

        def train():
            model.train()
            total_loss, total_accuracy = 0, 0
            total_preds = []
            for step, batch in enumerate(train_dataloader):
                if step % 50 == 0 and not step == 0:
                    print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
                # batch = [r.to(device) for r in batch]
                sent_id, mask, labels = batch
                model.zero_grad()
                preds = model.forward(sent_id, mask)
                loss = cross_entropy(preds, labels)
                total_loss = total_loss + loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # preventing the expolding gradient problem
                optimizer.step()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)
            avg_loss = total_loss / len(train_dataloader)
            total_preds = np.concatenate(total_preds, axis=0)
            return avg_loss, total_preds

        def evaluate():
            print("\nEvaluating...")
            model.eval()
            total_loss, total_accuracy = 0, 0
            total_preds = []
            for step, batch in enumerate(val_dataloader):
                if step % 50 == 0 and not step == 0:
                    print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
                # batch = [t.to(device) for t in batch]
                sent_id, mask, labels = batch
                with torch.no_grad():
                    preds = model(sent_id, mask)
                    loss = cross_entropy(preds, labels)
                    total_loss = total_loss + loss.item()
                    preds = preds.numpy()
                    total_preds.append(preds)
            avg_loss = total_loss / len(val_dataloader)
            total_preds = np.concatenate(total_preds, axis=0)
            return avg_loss, total_preds

        best_valid_loss = float('inf')
        train_losses = []
        valid_losses = []
        for epoch in range(epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
            train_loss, _ = train()
            valid_loss, _ = evaluate()
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')
        joblib.dump(model,'/Users/johnpoh/Desktop/bert/bert-fine-tuning.pkl')


    def predict_fine_tuning(self,contents):
        pretrain_model_name = 'voidful/albert_chinese_tiny'
        tokenizer = BertTokenizerFast.from_pretrained(pretrain_model_name)
        model = joblib.load('/Users/johnpoh/Desktop/bert/bert-fine-tuning.pkl')
        tokens = tokenizer.batch_encode_plus(
            contents,
            max_length=25,
            padding=True,
            truncation=True
        )

        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])

        with torch.no_grad():
            preds = model(input_ids, attention_mask)
            preds = preds.numpy()
            for pred in preds:
                arg = np.argmax(pred)
                if arg == 0:
                    print('negative')
                if arg == 1:
                    print('positive')


    def LabelBinarizer(self,tags):
        lb = preprocessing.LabelBinarizer()
        tags = lb.fit_transform(tags)
        labels = []
        print(lb.classes_)
        for index,label in enumerate(tags):
            labels.append(label)
        return labels

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(312, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask).pooler_output
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
