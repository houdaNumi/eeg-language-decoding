from load_data import data
import numpy as np
import csv



vect_feature=[]
for i in range(400):
    for word in data['sentenceData'][0,i]['word'][0]:

            t1= word['TRT_t1']
            t2 = word['TRT_t2']
            t3 = word['TRT_b1']
            t4 = word['TRT_b2']
            t5 = word['TRT_g1']
            t6 = word['TRT_g2']
            t7= word['TRT_a1']
            t8 = word['TRT_a2']

            if all(t.size > 0 for t in [t1,t2,t3,t4,t5,t6,t7,t8]):
                vect_feature.append(np.concatenate([t1,t2,t3,t4,t5,t6,t7,t8]))





X = np.array(vect_feature)
X = X.reshape(5110, -1)
print(X.shape)

sentiment_dict={}
with open('sentiment_labels_task1.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file ,delimiter=';')


        for line in csv_reader:
                if len(line) > 3 and line[1] and line[3]:
                        sentiment_dict[line[1]] = line[3]
        #for key, value in sentiment_dict.items():
               # print(f"{key} : {value}")










x_phrases=[]
y_labels=[]
for phrase in range(400):
        content = data['sentenceData'][0, phrase]['content'][0]
        label = sentiment_dict.get(content, None)
        if label is not None:
                for word in data['sentenceData'][0, phrase]['word'][0]:
                        t1 = word['TRT_t1']
                        t2 = word['TRT_t2']
                        t3 = word['TRT_b1']
                        t4 = word['TRT_b2']
                        t5 = word['TRT_g1']
                        t6 = word['TRT_g2']
                        t7 = word['TRT_a1']
                        t8 = word['TRT_a2']



                        if all(t.size > 0 for t in [t1, t2, t3, t4, t5, t6, t7, t8]):
                                x_phrases.append(np.concatenate([t1, t2, t3, t4, t5, t6, t7, t8]))
                                y_labels.append(label)





X = np.array(x_phrases)
y = np.array(y_labels)
X = X.reshape(X.shape[0], -1)
print(X.shape)
print(y.shape)

#stocker les labels et les phrases pour les utiliser pour le model
np.save("phrases.npy", X)
np.save("labels.npy", y)
X = np.load("phrases.npy")
y = np.load("labels.npy")
print(X.shape)
print(y.shape)