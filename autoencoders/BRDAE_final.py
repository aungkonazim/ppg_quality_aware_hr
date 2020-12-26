import pickle
X_acl,X_ppg,y,y_participant = pickle.load(open('../data/tabular_data.p','rb'))
from sklearn.preprocessing import RobustScaler,MinMaxScaler
for k in range(X_ppg.shape[0]):
    X_ppg[k] = MinMaxScaler().fit_transform(X_ppg[k])

import numpy as np
X_ppg = X_ppg[:,np.arange(0,512,2),:]

X_acl = np.concatenate([X_ppg,X_acl],axis=-1)

y = X_ppg[:,:,0].reshape(-1,256,1)
X_acl = X_acl[:,:,:1]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,LeaveOneGroupOut,LeavePGroupsOut
from sklearn.metrics import accuracy_score
from keras.layers import *
from keras.models import *
from keras import regularizers
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Conv1D,BatchNormalization,Dropout,Input,MaxPooling1D,Flatten,Dense,Input, GaussianNoise
from keras.models import Model, Sequential


logo = LeaveOneGroupOut()
logo = LeavePGroupsOut(n_groups=5)
for train_index, test_index in logo.split(X_acl, y, y_participant.reshape(-1)):
    train_x, test_x = X_acl[train_index], X_acl[test_index]
    train_y, test_y = y[train_index], y[test_index]
    train_participant, test_participant = y_participant[train_index], y_participant[test_index]
    break
print(train_x.shape,train_y.shape)

pickle.dump([train_participant,test_participant],open('../data/participant_split_for_BRDAE.p','wb'))
from joblib import Parallel,delayed

def random_noise(x):
    return x+(1/3)*np.random.normal(0,1,x.shape)

def sloping_noise(x):
    x = x.copy()
    def corrupt_single(xx):
        xx_temp = np.random.uniform(-1,1)
        xx_temp = np.array([xx_temp]*256).reshape(256,1)
        index = np.arange(0,256,1).reshape(256,1)
        return xx+(4/250)*np.multiply(index,xx_temp)

    for i in range(x.shape[0]):
        x[i] = corrupt_single(x[i])
    return x


def saturation_noise(x):
    x = x.copy()
    for i in range(x.shape[0]):
        low = np.random.randint(0,216)
        high = np.random.randint(low,min(256,np.random.randint(low+1,low+51)))
        x[i][low:high,0] = np.random.randint(0,2)
    return x

def generate_combinations(x):
    functions = [random_noise,saturation_noise,sloping_noise]
    indicator = np.random.randint(0,2,3)
#     indicator = [1,1,1]
    for j,i in enumerate(indicator):
        if i==1:
            x = functions[j](x)
    return x


def generate_corrupt_all(train_x,n):
    X = Parallel(n_jobs=20,verbose=2)(delayed(generate_combinations)(train_x.copy()) for i in range(n))
    return np.concatenate(X)

n = 50
train_y = np.concatenate([train_x]*n)
train_x = generate_corrupt_all(train_x,n=n)
 

train_x, val_x, train_y, val_y = train_test_split(train_x,train_y,test_size = 0.2,random_state=42)

print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape),(test_x.shape,test_y.shape))

def get_model_LSTM(input_shape=(256,1),act='tanh',loss="mse",opt='adam',n_classes=1):
    model = Sequential()
    model.add(GaussianNoise(stddev=1))
    model.add(Bidirectional(LSTM(40,return_sequences=True,
                      activation='tanh',input_shape=input_shape)))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss=loss,optimizer=opt)
    return model

model = get_model_LSTM()

from keras.models import load_model
filepath = '../model_files/BRDAE_final.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=20)
callbacks_list = [es,checkpoint]
history = model.fit(train_x,train_y,validation_data=(val_x,val_y), epochs=00, batch_size=1000,
          callbacks=callbacks_list,shuffle=True)



