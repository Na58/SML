import numpy as np
import features as fts
from sklearn.model_selection import train_test_split
COUNT = 100

def Read_From_train():
    relation = {}
    with open("./data/train.txt") as f:
        for line in f:
            rel = line.split("\t")
            source = rel[0].strip()
            target = rel[1:]
            relation[int(source)] = [int(i) for i in target]
    return relation

def Feature_From_Data():
    relation = Read_From_train()
    print("Start building features")
    InputData = []
    Labels = []
    ids = list(relation.keys())
    n = 50000
    step = int(20000/np.sqrt(n/2))
    for key,value in relation.items():
        for v in value:
            if n > 0:
                traninData = fts.Feature_Selector(key, v, relation)
                InputData.append(traninData)
                Labels.append(1)
                n -= 1
            else:
                break
        if n < 0:
            break

    for i in range(0,20000,step):
        for j in range(0,20000,step):
            if i != j:
                traninData = fts.Feature_Selector(ids[i],ids[j],relation)
                InputData.append(traninData)

                label = 1 if i in relation[ids[j]] else 0
                Labels.append(label)
    return InputData,Labels

def FeatureData_To_File(InputData,Lable,filename = "./data/features.txt"):
    with open(filename,'w') as f:
        for i in range(len(InputData)):
            f.write(str(Lable[i]))
            f.write("\t")
            for j in InputData[i]:
                f.write(str(j))
                f.write("\t")
            f.write("\n")

def FeatureData_From_File(filename = "./data/features.txt"):
    train = []
    lable = []

    with open(filename,'r') as f:
        for line in f:
            args = line.split("\t")[:-1]
            lable.append(float(args[0]))
            train.append([float(j) for j in args[1:]])
    return train,lable

def FeatureData_For_Kaggle(split):
    training_features = []
    training_lables = []
    test_features =  []
    test_lables = []
    pos = 0
    neg = 0

    # fe = open('testfeature.txt','w')
    #
    # with open("training_features.txt",'r') as f:
    #     for line in f:
    #         fet, lable = line.split(",")
    #         fetures = np.fromstring(fet,dtype=float,sep=' ')
    #         fetures = trick_with_data(fetures)

            # for i in fetures:
            #     fe.write(str(i))
            #     fe.write('\t')
            # fe.write("\n")

            # training_features.append(fetures)
            # training_lables.append(int(lable.strip()))

    # fe.close()


    #train = open("./data/log_training.csv",'a')
    # for i in range(48):
    #     train.write("F"+str(i+1))
    #     train.write(",")
    # train.write("Lable")
    # train.write("\n")

    for i in range(12):
        for prefix in ["Sink_Processing_by_cpu_","Source_Processing_by_cpu_"]:
            with open("./data/Confidential/"+prefix+str(i)+".txt",'r') as f:
                for line in f:
                    fet, lable = line.split(",")
                    # if int(lable) == 0 and neg > 67134:
                    #     continue
                    #
                    # if int(lable) == 0:
                    #     neg += 1
                    # else:
                    #     pos += 1


                    #fetures = np.fromstring(fet, dtype=float, sep=' ')
                    fetures = [float(item) for item in fet.split(" ")[:-1]]
                    fetures = trick_with_data(fetures)

                    # if i != 5:
                    training_features.append(fetures)
                    training_lables.append(int(lable.strip()))
                    # else:
                    #     test_features.append(fetures)
                    #     test_lables.append(int(lable.strip()))
                    # for j in fetures:
                    #     train.write(str(j))
                    #     train.write(",")
                    # train.write(str(lable))

    #train.close()

    #,test_features,test_lables
    print(pos,neg)

    training_features = normailize(training_features)
    training_features,test_features,training_lables,test_lables = train_test_split(training_features,training_lables,test_size=split)

    return training_features,training_lables,test_features,test_lables

def trick_with_data1(features):

    for i in range(3):

        SORENSENin = 2 * np.sqrt(features[i + 16 + 4]+1) / np.sqrt(features[i * 16 + 0] + features[i * 16 + 1]+1)
        SORENSENout = 2 * np.sqrt(features[i + 16 + 12]+1) / np.sqrt(features[i * 16 + 8] + features[i * 16 + 9]+1)

        features[i*16+0] = np.sqrt(features[i*16+0]+1)
        features[i*16+8] = np.sqrt(features[i*16+8]+1)

        features[i*16+1] = np.sqrt(features[i*16+1]+1)
        features[i*16+9] = np.sqrt(features[i*16+9]+1)

        features[i*16+2] = np.sqrt(features[i*16+2]+2)
        features[i*16+10] = np.sqrt(features[i*16+10]+2)

        features[i*16+3] = np.sqrt(abs(features[i*16+3])+2)
        features[i*16+11] = np.sqrt(abs(features[i*16+11])+2)

        HDIin = np.sqrt(features[i * 16 + 4]) / max(features[i * 16 + 0], features[i * 16 + 1])
        HDIout = np.sqrt(features[i * 16 + 12]) / max(features[i * 16 + 8], features[i * 16 + 9])

        features.extend([HDIin, HDIout, SORENSENin, SORENSENout])

    newfeature = []
    global COUNT
    for i in range(len(features)):
        if i in [COUNT,COUNT+10,COUNT+20,COUNT+30,COUNT+40,COUNT+50]:
            newfeature.append(features[i])

    return np.array(newfeature)

def trick_with_data(features):
    # features[7] = np.log10(features[0]+1) * np.log10(features[1]+1)
    # features[15] = np.log10(features[8]+1) * np.log10(features[9]+1)

    for i in range(3):
        pass
        # union = features[i*26+0] + features[i*26+1] - features[i*26+4]
        # if union == 0:
        #     features[i*26+5] = 0
        # else:
        #     features[i*26+5] = features[i*26+4] / union
        #
        # union1 = features[i * 26 + 8] + features[i * 26 + 9] - features[i * 26 + 12]
        # if union1 == 0:
        #     features[i*26+13] = 0
        # else:
        #     features[i * 26 + 13] = features[i * 26 + 12] / union1
        #
        # try:
        #     features[i*26+6] = features[i*26+4] / (features[i*26+0]*features[i*26+1])
        # except:
        #     features[i*26+6] = 0
        #
        # try:
        #     features[i*26+14] = features[i*26+12] / (features[i*26+8]*features[i*26+9])
        # except:
        #     features[i*26+14] = 0
        #

    #newfeature = features
    newfeature = []

    for i in range(len(features)):
 #       x - y
        if i % 13 == 3: #in [3,16,29,42,55,68]:
            continue
 #       x * y
        elif i % 13 == 7: #in [7,20,33,46,59,72]:
            continue
 #       x
        elif i % 13 == 0: #in [0,13,26,39,52,65]:
            continue
 #       cos
        elif i % 13 == 8: #in [8,21,34,47,60,73]:
            continue
        # #HPI
        # elif i % 13 == 10: #in [10,23,36,49,62,75]:
        #     continue
        # #Sorensen
        # elif i % 13 == 11: #in [11,24,37,50,63,76]:
        #     continue
        # y
        elif i % 13 == 1:
            continue
        elif i % 13 == 2:
            continue
        else:
        # if i % 13 == 12:
             newfeature.append(features[i])

    return np.array(newfeature)

def normailize(feature_matrix):
    matrix = np.matrix(feature_matrix)
    max = np.amax(matrix,axis=0)
    min = np.amin(matrix,axis=0)
    dlt = max - min
    final_matrix = (matrix - min) / dlt
    return final_matrix.tolist()


if __name__ == '__main__':
    FeatureData_For_Kaggle()
