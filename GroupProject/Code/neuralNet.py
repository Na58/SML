from sklearn.neural_network import MLPClassifier,MLPRegressor
from file import Feature_From_Data,FeatureData_To_File,FeatureData_From_File
from file import FeatureData_For_Kaggle
import numpy as np
import random
from random import seed

#keras library
from keras import models
from keras import layers
from keras.layers import Dense,InputLayer,Dropout
import keras
import keras.backend as K
from keras import regularizers
import tensorflow as tf

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class Trainer:

    def __init__(self):
        #,self._test_input,self._test_lable
        self._train_input,self._train_lable,t_festures,t_labels = FeatureData_For_Kaggle(0.1)
        self._test_input,self._test_lable = self.get_test_data(1,feature=t_festures,lable=t_labels)
        #self._test_input,self._test_lable = self.get_test_data(0,feature=t_festures,lable=t_labels)
        #random.seed(900)

    def Training_With_RandomForest(self):
        rf = RandomForestClassifier(max_depth=None,min_samples_split=2,verbose=0,max_features=None,oob_score=True)
        train_input,train_label = self.get_data()
        print("Start training data")
        rf.fit(train_input,train_label)
        print("Start predict data")

        #self.test_manully(rf,0)
        #self.test_cross_validation(rf)
        self.test_auc(rf)
        return rf

    def Training_With_Sklearn_MLP(self,wtest):
        clf = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(50,50,50),alpha=0.001,max_iter=300,batch_size=150)
        #clf = MLPRegressor(solver='adam', activation='logistic', hidden_layer_sizes=(24,24,24),alpha=0.01,max_iter=300)

        train_input,train_lable = self.get_data()

        if wtest:
            max = 0
            arg = 7
            for i in range(7,31):
                clf.hidden_layer_sizes = (i,i)
                print("Start training data")
                clf.fit(train_input, train_lable)
                print("Start predict data")
                acc = self.test(clf,1)
                if acc > max:
                    max = acc
                    arg = i
            print(max,arg)
        else:
            print("Start training data")
            clf.fit(train_input, train_lable)
            print("Start predict data")
            #self.test(clf, 1)
            self.test_auc(clf)
            return clf

    def Training_With_Keras(self,units):

        model = keras.Sequential()
        #input = Dense(7,activation='tanh',kernel_regularizer=regularizers.l1(1))
        #kernel_regularizer = regularizers.l1(0.001)
        model.add(Dense(units, input_dim=42, activation=K.relu))
                        # kernel_regularizer=regularizers.l2(0.1)))
        # model.add(Dropout(0.4))
        model.add(Dense(units, activation=K.relu))
        # model.add(Dropout(0.4))
        model.add(Dense(units, activation=K.relu))
        # model.add(Dropout(0.4))
#        model.add(Dense(units, activation=K.relu))
        # model.add(Dropout(0.4))
        model.add(Dense(1, activation=K.sigmoid))

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'])

        train_input, train_lable = self.get_data()

        print("Start training data")
        history = model.fit(train_input, train_lable,epochs=40,batch_size=200,verbose=1,validation_split=0.2)
        loss = history.history['loss'][-1]

        #result = self.test_sklearn(model,1)
        result = self.test_auc(model)
        return model

    def custom_roc(self,y_ture,y_pred):
        from sklearn import metrics

        ture = np.array(y_ture)
        pre = np.array(y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(ture, pre, pos_label=1)
        acc = metrics.auc(fpr, tpr)
        return acc

    def grid_search_keras(self):
        for i in range(4,64,4):
            print(">>>> Search with units: ", i)
            self.Training_With_Keras(i)


    def Training_With_Logistic(self):
        # model = keras.Sequential()
        # model.add(Dense(1,input_dim=7,activation='sigmoid'))
        # model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        # train,lable = self.get_data()
        # model.fit(train,lable,epochs=1,verbose=0)
        # self.test(model,0)

        train_input, train_lable = self.get_data()
        model = LogisticRegression(solver='newton-cg',max_iter=300,verbose=1,penalty='l2')
        model.fit(train_input, train_lable)
        result = self.test_auc(model)
        # if result > 1:
        #     from sklearn.externals import joblib
        #     joblib.dump(model,"./data/logis_model.pkl")
        #     return 1
        # return -1
        return model

    def get_data(self):
        print("Start reading data")

        # train_input, train_lable = Feature_From_Data()
        ## Add feature to file
        # FeatureData_To_File(train_input,train_lable)


        train_input = np.matrix(self._train_input)
        train_lable = np.matrix(self._train_lable).transpose()

        return train_input,train_lable

    def get_test_data(self,tag,**kwargs):
        test_input = []
        test_label = []

        if tag == 0:
            _test_input = kwargs['feature']
            _test_label = kwargs['lable']
            count = 0
            for i in range(5000):
                idx = random.randrange(len(_test_input))
                if _test_label[idx] == 1:
                    continue
                if count > 1000:
                    break
                # if self._train_lable[idx] == 0 and len(test_label) < 1000 and self._train_input[idx] not in test_input:
                test_input.append(_test_input[idx])
                test_label.append(_test_label[idx])
                count += 1

        else:
            test_input = kwargs['feature']
            test_label = kwargs['lable']



        test_input = np.matrix(test_input)
        test_label = np.matrix(test_label).transpose()

        return test_input,test_label


    # Code for test
    def test(self,model,type):
        if type == 0:
            score = model.evaluate(self._test_input,self._test_lable,batch_size=128)
            print(score)
        else:
            score = model.score(self._test_input,self._test_lable)
            print(score)

        return score

    def test_cross_validation(self,model):
        scores = cross_val_score(model,self._test_input,self._test_lable,cv=5)
        print('Acc: ',scores.mean())

    def test_manully(self,model,type):
        r = 0
        for idx in range(self._test_input.shape[0]):
            if type == 0:
                pre = model.predict(self._test_input[idx,:])[0]
            else:
                pre = model.predict(self._test_input[idx,:])[0][0]
            #print(model.predict_proba(self._test_input[idx, :]))
            act = self._test_lable[idx, 0]
            equal = round(pre) == act
            if equal:
                r += 1
        print("Acc:", r / len(self._test_input))
        return r/len(self._test_input)

    def test_auc(self,model):
        from sklearn import metrics
        from TestCSV import plotFromList

        pre = model.predict(self._test_input)
        # if isinstance(model,LogisticRegression):
        #     loss = 0
        # else:
        #     loss = model.evaluate(self._test_input,self._test_lable,batch_size=100)
        fpr, tpr,thresholds = metrics.roc_curve(self._test_lable,pre,pos_label=1)
        acc = metrics.auc(fpr,tpr)
        #print("Loss: ", loss)
        print("Auc: ", acc)
        #plotFromList(pre, self._test_lable)

# Code for generating test result
def test_feature(model):
    from file import trick_with_data


    test_feature_data = []
    with open("./data/Confidential/test_feature.txt", 'r') as f:
        for line in f:
            fet, lable = line.split(",")
            fetures = [float(item) for item in fet.split(" ")[:-1]]
            test_feature_data.append(fetures)


    # result = []
    # for fet in test_feature_data:
    #     tricked_data = trick_with_data(fet)
    #     #tricked_data = fet
    #     tmp = model.predict(np.matrix(tricked_data))[0][0]
    #     #tmp = model.predict(np.matrix(tricked_data))[0]
    #     tmp = np.around(tmp,decimals=8)
    #     result.append(tmp)
    #
    from file import normailize
    nfeature = [trick_with_data(i) for i in test_feature_data]
    nfeature = normailize(nfeature)
    result = model.predict(np.matrix(nfeature))
    if not isinstance(model,keras.Sequential):
        result = [result[i] for i in range(len(result))]
    else:
        result = [result[i,0] for i in range(result.shape[0])]

    with open("./data/test.csv",'w') as csv:
        csv.write("Id,Prediction\n")
        for i in range(len(result)):
            csv.write(str(i+1))
            csv.write(",")
            #csv.write('{:0.20}'.format(result[i]))
            csv.write(str(result[i]))
            csv.write("\n")

    from TestCSV import plot
    plot("./data/test.csv")

if __name__ == '__main__':
    trainer = Trainer()


    while True:
        #unit = 56
        model = trainer.Training_With_Keras(20)
        #model = trainer.Training_With_Logistic()
        #model = trainer.Training_With_RandomForest()
        test_feature(model)
        break

        #trainer.grid_search_keras()

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #
    # print("Train with sklearn nerual network")
    # trainer.Training_With_sklearn(False)
    #
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    #
    # print("Train with logistic regression")
    # trainer.Training_With_Logistic()