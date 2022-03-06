from __future__ import unicode_literals
import os
os.environ['PYTHONHASHSEED']= '0'
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import random as rn
rn.seed(0)
path = 'C:/Users/Kavishka/anaconda3/Library/share/fonts/Nirmala.ttf'
prop = font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = prop.get_name()

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV,train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.random.set_random_seed(0)
import tensorflow.keras.backend as K
#from keras import layers
from sklearn import metrics
#from keras import regularizers
#from keras import models,optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.layers import Dense,Input,SpatialDropout1D,Dropout,Flatten, SimpleRNN,LSTM,RNN,GRU
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import re
import gensim
from gensim.models import word2vec
import gensim.models as FastText
from gensim.test.utils import datapath
import eli5
from eli5.lime import TextExplainer
import shap
from lime.lime_text import LimeTextExplainer
from IPython.display import display
pd.set_option('display.max_colwidth', 1000)
# import helper function script
import sys
sys.path.insert(1,'G:\\Github\\Sinhala-Hate-Speech-Detection')
import utills
import sinhala_stemmer
import neptune
from neptunecontrib.monitoring.metrics import expand_prediction, log_class_metrics, log_binary_classification_metrics, log_classification_report,log_confusion_matrix,log_prediction_distribution
from neptunecontrib.api import log_table,log_chart
from neptunecontrib.monitoring.keras import NeptuneMonitor
from dotenv import load_dotenv

load_dotenv()
NEPTUNE_PROJECT= os.getenv('NEPTUNE_PROJECT')
NEPTUNE_API_TOKEN = os.getenv(('NEPTUNE_API_TOKEN'))
neptune.init(project_qualified_name= NEPTUNE_PROJECT,api_token=NEPTUNE_API_TOKEN) 
     
     

class TextClassifier():
    def __init__(self,tag ="DL model",EMBEDDING=None,EPOCHS=50,BATCH_SIZE=16,MAX_SEQ_LEN=100,EMBEDDING_SIZE=300,LEN_VOCAB = 20000,lr=0.001,trainable= False):
        self.EMBEDDING = EMBEDDING
        self.LEN_VOCAB = LEN_VOCAB
        self.EMBEDDING_SIZE =EMBEDDING_SIZE
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS =EPOCHS
        self.lr = lr
        self.tag = tag
        self.trainable = trainable
        self.unit = 64
        self.layer = 1
        self.dropout_rate =0.3
        self.recurr_dropout = 0.0
        self.activation = 'relu'
        self.df_A = self.load_data()
        self.X_tr, self.X_te, self.Y_train, self.Y_test = train_test_split(self.df_A['cleaned'], self.df_A['label'], test_size=0.3, random_state=0, stratify=self.df_A['label'].values)
        print("X train {} Y train {} X test {} Y test {}".format(self.X_tr.shape, self.Y_train.shape, self.X_te.shape, self.Y_test.shape))
        if(EMBEDDING != None):
            self.vocab = self.build_vocab()
            self.emb_model = self.load_emb_model()
            self.embeddings_index = self.get_emb_index()
            self.oov_words = self.check_coverage()
            if(EMBEDDING=="fasttext"):
                self.fasttext_OOV()
            else:
                self.add_stem()
            self.oov_words = self.check_coverage()
        self.token = Tokenizer(num_words=self.LEN_VOCAB)
        self.X_train ,self.X_test, self.word_index = self.create_sequence()
        if(EMBEDDING != None):
            self.emb_matrix = self.embedding_matrix()
        self.model = None
        self.arr_index = None
        self.hist = None
        self.Y_pred = None
        
        
    
    def load_data(self):
        print("Load data\n")
        path = 'G:/Github/Sinhala-Hate-Speech-Detection/Datasets/processed/no_stemming/df_A.csv'
        df_A = pd.read_csv(path)  
        df_A.drop(index =[610,3070],inplace=True)
        return df_A

    def load_emb_model(self):
        print("Load Embedding model\n")
        if self.EMBEDDING == "w2v_skipgram":
            model = word2vec.Word2Vec.load("G:/Github/Sinhala-Hate-Speech-Detection/Embedding_models/word2vec/word2vec_300.w2v")
        elif self.EMBEDDING == "w2v_cbow":
            model = word2vec.Word2Vec.load("G:/Github/Sinhala-Hate-Speech-Detection/Embedding_models/CBOW-word2vec/cbow_300.w2v")
        elif self.EMBEDDING == "fasttext":
            #FastText.load_fasttext_format("../../../corpus/analyzed/saved_models/wiki.si.bin")
            model = FastText.fasttext.load_facebook_model(datapath("G:/Github/Sinhala-Hate-Speech-Detection/Embedding_models/cc.si.300.bin"))
            #model = word2vec.Word2Vec.load("G:/Github/Sinhala-Hate-Speech-Detection/Embedding_models/fasttext_300.w2v")
        else:
            print("Invalid argument. Need w2v_skipgram or w2v_cbow or fasttext as argument")
            model = None
        return model

    def get_emb_index(self):
        print("Create embedding index\n")
        from gensim.models import word2vec
        embeddings_index ={}
        for index, word in enumerate(self.emb_model.wv.index_to_key):
            embeddings_index[word] = self.emb_model.wv.get_vector(word)
        print('found %s word vectors' % len(embeddings_index))
        return embeddings_index

    def build_vocab(self):
        print("Build vocab\n")
        sentences = self.X_tr.apply(lambda x: x.split()).values
        vocab = {}
        for sentence in sentences:
            for word in sentence:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
        return vocab

    def check_coverage(self):
        print("Check coverage")
        import operator
        known_words = {}
        unknown_words = {}
        nb_known_words = 0
        nb_unknown_words = 0
        for word in self.vocab.keys():
            try:
                known_words[word] = self.embeddings_index[word]
                nb_known_words += self.vocab[word]
            except:
                unknown_words[word] = self.vocab[word]
                nb_unknown_words += self.vocab[word]
                pass

        print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(self.vocab)))
        print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
        unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

        return unknown_words


    def fasttext_OOV(self):
        count = 0
        for word, freq in self.oov_words:
            if word in self.emb_model.wv:
                self.embeddings_index[word] = self.emb_model.wv.get_vector(word)
                count+=1
        print(f"Added {count} words to embedding with rare words handling of fasttext")
        #return embedding_index  

        
    def add_stem(self):
        
        import sinhala_stemmer
        word_len = 1
        stemmer = sinhala_stemmer.SinhalaStemmer()
        stem_count = [0,0]
        word_list = {}

        for word,freq in self.vocab.items():
            word_ls = stemmer.stem(word, True, word_len)[0]   
            word_ss = stemmer.stem(word, False, word_len)[0]  

            # longer suffix     
            if word not in self.embeddings_index and word_ls  in self.embeddings_index:
                self.embeddings_index[word] = self.embeddings_index[word_ls]
                stem_count[0]+=1
                word_list[word]=[word_ss,word_ls] 

            # shorter suffix
            elif word not in self.embeddings_index and word_ss in self.embeddings_index: 
                self.embeddings_index[word] = self.embeddings_index[word_ss]
                stem_count[1]+=1 
                word_list[word]=[word_ls,word_ss]  
        print(f"Added {stem_count[0]} words to embedding with longer suffix")
        print(f"Added {stem_count[1]} words to embedding with shorter suffix")
        print(len(self.embeddings_index))  
        #print(word_list)  
        #return embedding      


    def create_sequence(self):

        #token = Tokenizer(num_words=self.LEN_VOCAB)
        self.token.fit_on_texts(self.X_tr)
        word_index = self.token.word_index
        print("dictionary size: ", len(word_index))

        # ensure equal length vectors 
        train_seq_x = sequence.pad_sequences(self.token.texts_to_sequences(self.X_tr), maxlen=self.MAX_SEQ_LEN)
        test_seq_x = sequence.pad_sequences(self.token.texts_to_sequences(self.X_te), maxlen=self.MAX_SEQ_LEN)
        return (train_seq_x,test_seq_x,word_index)

    def embedding_matrix(self):
        print("Embedding matrix \n")
        all_embs = np.stack(self.embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        #embed_size = all_embs.shape[1]
        word_index = self.word_index
        embedding_matrix = np.random.normal(emb_mean, emb_std, (self.LEN_VOCAB, self.EMBEDDING_SIZE))
        
        for word, i in word_index.items():
            if i >= self.LEN_VOCAB:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
        del self.emb_model
        return embedding_matrix

    
    # def get_model(self,model_type):
    #     print("Build ML model")
    #     model = Sequential()
    #     if(self.EMBEDDING== None):
    #         model.add(Embedding(output_dim=self.EMBEDDING_SIZE, 
    #                             input_dim=self.LEN_VOCAB, 
    #                             input_length=self.MAX_SEQ_LEN,
    #                            # weights=[self.emb_matrix], # Additionally we give the Wi
    #                             trainable=self.trainable))
        
    #     else:
    #         model.add(Embedding(output_dim=self.EMBEDDING_SIZE, 
    #                             input_dim=self.LEN_VOCAB, 
    #                             input_length=self.MAX_SEQ_LEN,
    #                             weights=[self.emb_matrix], # Additionally we give the Wi
    #                             trainable=self.trainable)) # Don't train the embeddings - just use GloVe embeddings
    #         # We can start with pre-trained embeddings and then fine-tune them using our data by setting trainable to True
    #     if(model_type=="RNN"):
    #         model.add(SimpleRNN(128, activation='relu',dropout=0.2, recurrent_dropout=0.3))
    #     elif(model_type == "LSTM"):
    #         model.add(LSTM(128, activation='relu',dropout=0.2, recurrent_dropout=0.3))
    #     elif(model_type == "GRU"):
    #         model.add(GRU(128, activation='relu',dropout=0.2, recurrent_dropout=0.3))
    #     elif(model_type == "BiLSTM"):
    #         model.add(Bidirectional(LSTM(128, activation='relu',dropout=0.2, recurrent_dropout=0.3)))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(1,activation='sigmoid'))
    #     optimizer_adam = Adam(learning_rate=self.lr)
    #     model.compile(loss='binary_crossentropy',
    #                 optimizer=optimizer_adam,
    #                 metrics=['acc'])
    #     print(model.summary())
    #     self.model = model
    #     return model

    def train_model(self,model):
  
        #define callbacks
        early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.01, patience=5, verbose=1,mode='min',restore_best_weights=True)
        checkpoints = ModelCheckpoint(filepath='G:/Github/Sinhala-Hate-Speech-Detection/trained_models/checkpoints/finetuned/'+self.tag+'/model.h5', monitor="val_loss", mode="min", verbose=1, save_best_only=True)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-5,mode='min', verbose=1)
        callbacks_list = [lr,early_stopping,NeptuneMonitor(),checkpoints] #,NeptuneMonitor(),checkpoints]

        #model training
        print("started training")
        hist = model.fit(self.X_train, self.Y_train, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, callbacks=callbacks_list,validation_split=0.1, shuffle=False, verbose=2)
        self.model = model
        self.hist = hist
        return model,hist

    def log_result(self,Y_pred):
        print("========= Eperiment - ",self.tag," =========")
        PARAMS = {'epoch': self.EPOCHS,
          'lr': self.lr,
          'batch':self.BATCH_SIZE,
          'embedding':self.EMBEDDING,
          'emb_trainable':self.trainable,
          'unit':self.unit,
          'layer':self.layer,
          'dropout_rate':self.dropout_rate,
          'recurr_dropout':self.recurr_dropout,
          'activation':self.activation

          }
        neptune.create_experiment(self.tag,params=PARAMS)
        neptune.append_tag(['finetuned experiment',self.tag])
        
        log_class_metrics(self.Y_test, Y_pred)
        log_confusion_matrix(self.Y_test, Y_pred)
        log_classification_report(self.Y_test,Y_pred)
        
    def model_evaluate(self,model,hist ):
        train_loss, train_acc = model.evaluate(self.X_train, self.Y_train,batch_size=self.BATCH_SIZE, verbose=1)
        print("train loss - ",train_loss," train acc- ",train_acc)
        test_loss, test_acc = model.evaluate(self.X_test,self.Y_test,batch_size=self.BATCH_SIZE, verbose=1)
        print("test loss - ",test_loss," test acc- ",test_acc)

        # plot loss during training
        #from matplotlib import pyplot
        print('Train acc: %.3f, Test acc: %.3f' % (train_acc, test_acc))
        plt.subplot(211)
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel("Cross Entropy Loss")
        plt.plot(hist.history['loss'],lw=2.0, color='b', label='train')
        plt.plot(hist.history['val_loss'],lw=2.0, color='r', label='validation')
        plt.legend(loc='upper right')
        # plot accuracy during training
        plt.subplot(212)
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(hist.history['acc'],lw=2.0, color='b', label='train')
        plt.plot(hist.history['val_acc'],lw=2.0, color='r', label='validation')
        plt.legend(loc='upper right')
        plt.show()
        #(model.predict(x) > 0.5).astype("int32")
        #np.where(y_pred > threshold, 1,0)
        Y_pred =(model.predict(self.X_test) > 0.5).astype("int32")# model.predict_class(test_seq)
        Y_pred = np.squeeze(Y_pred)
        self.log_result(Y_pred)
        neptune.log_metric("train_loss",train_loss)
        neptune.log_metric("train_acc",train_acc)
        neptune.log_metric("test_loss",test_loss)
        neptune.log_metric("test_acc",test_acc)

        print(classification_report(self.Y_test,Y_pred))
        #Y_pred = np.squeeze(Y_pred)
        utills.confusion_Matrix(self.Y_test,Y_pred)
        auc_score = utills.PlotRocAuc(self.Y_test,Y_pred,'blue',self.tag)
        neptune.log_metric("auc_score",auc_score)
        self.Y_pred = Y_pred
        return Y_pred

    def result_map(self,x):
        if(x==0):
            return "Not Hate"
        if(x==1):
            return "Hate"

    def save_predictions(self,Y_pred,filename):
        i=0
        arr_index=self.X_te.index
        Results = pd.DataFrame(columns=["org_index","comment","pred_label","label"])
        for s in self.X_te:
            print(s)
            Results.at[i,"org_index"] = arr_index[i]
            Results.at[i,"comment"] = s
            Results.at[i,'pred_label'] = Y_pred[i]
            Results.at[i,'label'] = self.Y_test[arr_index[i]]
            print("Predicted Label : ",self.result_map(Y_pred[i])," | Turth Label : ",self.result_map(self.Y_test[arr_index[i]]))
            i+=1
            print()
        Results.to_csv("G:/Github/Sinhala-Hate-Speech-Detection/DL_models/Predictions_result/finetuned/"+filename+".csv",index=False)
        print(Results.head(n=10))

    # def predict_proba(self,arr):
 
    #     pred=self.model.predict(sequence.pad_sequences(self.token.texts_to_sequences(arr),maxlen=self.MAX_SEQ_LEN))
    #     returnable=[]
    #     for i in pred:
    #         temp=i[0]
    #         returnable.append(np.array([1-temp,temp]))
    #     return np.array(returnable)
   

    # def error_analysis(self,model,Y_pred):
    #     #token = Tokenizer(num_words=self.LEN_VOCAB)
    #     #token.fit_on_texts(self.X_tr)
    #     lime_explainer= LimeTextExplainer(class_names=[0,1])
    #     te = TextExplainer(random_state=0)
    #     distrib_samples = self.X_train[:100]
    #     explainer = shap.DeepExplainer(model, distrib_samples)
    #     # explain the first 25 predictions
    #     # explaining each prediction requires 2 * background dataset size runs
    #     num_explanations = 50
    #     shap_values = explainer.shap_values(self.X_test[:num_explanations])
    #     shap.initjs()
    #     num2word = {}
    #     arr_index=self.X_te.index
    #     for w in self.word_index.keys():
    #         num2word[self.word_index[w]] = w
    #     x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), self.X_test[i]))) for i in range(num_explanations)])
    #     i=0
    #     for s in self.X_te:
    #         if(i==25):
    #             break
    #         if(self.Y_test[arr_index[i]] != Y_pred[i]):
    #             print(s)
    #             print("Predicted Label : ",self.result_map(Y_pred[i])," | Turth Label : ",self.result_map(self.Y_test[arr_index[i]]))
    #             te.fit([self.X_te[arr_index[i]]],self.predict_proba)
    #             display(te.show_prediction(target_names=[0,1]))
    #             display(lime_explainer.explain_instance(self.X_te[arr_index[i]],self.predict_proba).show_in_notebook(text=True))
    #             shap.force_plot(explainer.expected_value[0], shap_values[0][i], x_test_words[i],matplotlib=True)
    #         i+=1
    #         print()
