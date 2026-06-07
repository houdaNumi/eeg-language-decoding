import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat("resultsZAB_SR.mat")
print(data.keys())

print(data['sentenceData'].shape)
print(data['sentenceData'][0,0]['content'][0])
print(data['sentenceData'][0,0]['word'].shape)


rawEeg= data['sentenceData'][0,0]['word'][0][0]['rawEEG'][0][0][0,:]
plt.plot(rawEeg)
plt.show()



print(data['sentenceData'][0,0]['word'][0].shape)
print(data['sentenceData'][0,0]['word'][0].dtype)
#print(data['sentenceData'][0,0]['word'][0]['TRT_t1'])


print(data['sentenceData'][0,0]['word'][0]['TRT_t1'][0])
