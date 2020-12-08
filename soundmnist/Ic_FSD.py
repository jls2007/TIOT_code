
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Input,Reshape,Activation
import numpy as np
import librosa
import os
from keras.utils import to_categorical
#from sklearn.metrics import classification_report
#import tensorflow as tf


def wav2mfcc(file_path, max_pad_len=45):
    wave, sr = librosa.load(file_path,mono=True) 
    #sr=22050
    
    mfcc = librosa.feature.mfcc(wave)

    if mfcc.shape[1]<=45:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc=mfcc[:,0:45]
        
    return mfcc

def get_data(): #get the wav data in /recordings and then convert them to spectrogram with labels

    labels = []
    mfccs = []

    for f in os.listdir('./recordings'):
        if f.endswith('.wav'):
            # MFCC
            mfccs.append(wav2mfcc('./recordings/' + f))

            # List of labels
            label = f.split('_')[0]
            labels.append(label)

    return np.asarray(mfccs), to_categorical(labels)

def get_all():
    mfccs, labels = get_data()

    dim_1 = mfccs.shape[1]
    
    dim_2 = mfccs.shape[2]
    
    channels = 1
    classes = 10

    X = mfccs
    X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
    y = labels

    input_shape = (dim_1, dim_2, channels)
    print(input_shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    model = get_cnn_model(input_shape, classes)
  
    return X_train, X_test, y_train, y_test, model



def get_cnn_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20,45,1)))


    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
 

    model.add(Conv2D(64, kernel_size=(3, 6), activation='relu'))
  

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model



def check_preds(X, y):

    trained_model = keras.models.load_model('./Ic_FSD.h5')
    
    loss,acc=trained_model.evaluate(X,y)

    print('test loss of infmodel Ic',loss)
    print('test accuracy of infmodel Ic',acc)
    
    #or you can use
    #predictions = trained_model.predict_classes(X)
    #print(classification_report(y, to_categorical(predictions)))



keras_callback_inf=[keras.callbacks.ModelCheckpoint(filepath='./Ic_FSD.h5',monitor='val_loss',save_best_only=True)]
X_train,X_test, y_train, y_test, cnn_model= get_all()

cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

cnn_model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=2, validation_split=0.1,callbacks=keras_callback_inf)
print(cnn_model.summary())
 
       
check_preds(X_test, y_test)

#you can use these codes to convert the first 10 original mfccs back to wav files
#for i in range(10): 
    #inv=librosa.feature.inverse.mfcc_to_audio(X_test[i,:,:,0])
    #librosa.output.write_wav('./inverse/inverse_result'+str(i)+'.wav',inv,sr=22050)
