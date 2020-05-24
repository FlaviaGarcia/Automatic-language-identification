import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
import soundfile as sf
from sidekit.frontend.features import plp, compute_delta, mfcc

class DataGenerator(keras.utils.Sequence):

    def __init__(self, root_path, batch_size=32, shuffle=True, net='cnn', feat='melspec', precomputed=False):

        self.batch_size = batch_size
        self.root_path = root_path
        self.shuffle = shuffle
        self.list_dir = []
        self.net = net
        self.precomputed = precomputed
        if self.net == 'dnn':
            self.batch_size = 75
        self.feat = feat
        if self.net == 'cnn':
            self.dim = (128, 79)
            if self.feat=='mfcc' or self.feat=='plp':
                self.dim = (39, 75)
            elif self.feat=='combined':
                self.dim1 = (39, 75)
                self.dim2 = (128, 79)
        self.n_channels = 1
        targets = os.listdir(self.root_path)
        self.target_to_class = {lang: i for i, lang in enumerate(targets)}
        self.n_classes = len(self.target_to_class)
        i = 0
        self.count = {}
        for target in targets:
            target_path = os.path.join(os.path.join(self.root_path, target), 'clips_cut/')
            self.count[target] = 0
            for path in os.listdir(target_path):
                if (not path.startswith('.')):
                    item_path = os.path.join(target_path, path)
                    self.list_dir.append(item_path)
                    self.count[target] += 1
                
            i += 1

        self.n_classes = len(targets)
        self.on_epoch_end()

    def getTargets(self):
        targets = []
        for item_path in self.list_dir:
            label = item_path.split('/')[-3]
            targets.append(self.target_to_class[label])
            
        return np.array(targets)

    def __len__(self):
        if (self.net == 'dnn'):
            return len(self.list_dir)
        return len(self.list_dir)//self.batch_size

    def __getitem__(self, index):
        
        # Generate indexes of the batch
        if (self.net == 'cnn'):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            list_dirs_temp = [self.list_dir[k] for k in indexes]
            # Generate data
            X, y = self.__data_generation_cnn(list_dirs_temp)
        elif (self.net == 'dnn'):
            pos = index%3
            start = index - pos
            indexes = np.roll(self.indexes[start:start+3], pos)
            list_dirs_temp = [self.list_dir[k] for k in indexes]
            X, y = self.__data_generation_dnn(list_dirs_temp)
        elif (self.net == 'lstm'):
            X, y = self.__data_generation_lstm(self.list_dir[self.indexes[index]])

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_dir))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation_cnn(self, list_dirs_temp):

        # Initialization
        y = np.empty((self.batch_size), dtype=int)
        if self.feat == 'combined':
            X1 = np.empty((self.batch_size, *self.dim1, self.n_channels))
            X2 = np.empty((self.batch_size, *self.dim2, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        labels = []
        for i, item_path in enumerate(list_dirs_temp):

            if self.feat == 'melspec':
                waveform, sr = sf.read(item_path)
                mel = librosa.feature.melspectrogram(waveform, sr=16000)
                ps_db = librosa.power_to_db(mel, ref=np.max).reshape((*self.dim, 1))
                X[i,] = (ps_db - np.mean(ps_db))/np.var(ps_db)
            elif self.feat == 'mfcc':
                if self.precomputed:
                    X[i,] = np.load(item_path)
                else:
                    waveform, sr = sf.read(item_path)
                    feat_item = mfcc(waveform+1e-9, maxfreq=sr/2.0, nwin=.128, shift=.032)[0]
                    feat_delta1 = compute_delta(feat_item)
                    feat_delta2 = compute_delta(feat_delta1)
                    feat = np.concatenate((feat_item, feat_delta1, feat_delta2), axis=1).T.reshape((*self.dim, 1))
                    X[i,] = (feat - np.mean(feat))/np.var(feat)
            elif self.feat == 'plp':
                if self.precomputed:
                    X[i,] = np.load(item_path)
                else:
                    waveform, sr = sf.read(item_path)
                    feat_item = plp(waveform+1e-9, fs=sr, rasta=False, nwin=.128, shift=.032)[0]
                    feat_delta1 = compute_delta(feat_item)
                    feat_delta2 = compute_delta(feat_delta1)
                    feat = np.concatenate((feat_item, feat_delta1, feat_delta2), axis=1).T.reshape((*self.dim, 1))
                    X[i,] = (feat - np.mean(feat))/np.var(feat)
            elif self.feat == 'combined':
                waveform, sr = sf.read(item_path)
                mel = librosa.feature.melspectrogram(waveform, sr=16000)
                ps_db = librosa.power_to_db(mel, ref=np.max).reshape((*self.dim2, 1))
                X2[i,] = (ps_db - np.mean(ps_db))/np.var(ps_db)
                feat_item = mfcc(waveform+1e-9, maxfreq=sr/2.0, nwin=.128, shift=.032)[0]
                feat_delta1 = compute_delta(feat_item)
                feat_delta2 = compute_delta(feat_delta1)
                feat = np.concatenate((feat_item, feat_delta1, feat_delta2), axis=1).T.reshape((*self.dim1, 1))
                X1[i,] = (feat - np.mean(feat))/np.var(feat)
            # Store class
            label = item_path.split('/')[-3]
            labels.append(item_path)
            y[i] = self.target_to_class[label]

        target = keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.feat == 'combined':
            return (X1, X2), target
        return X, target
    
    def __data_generation_dnn(self, list_dirs_temp):

        # Initialization
        X = np.empty((self.batch_size, 39*21))
        target = np.empty((self.batch_size, 8))
        for i, item_path in enumerate(list_dirs_temp):
            if self.feat=='mfcc':
                if self.precomputed:
                    feat = np.load(item_path)[:,:,0].T
                else:
                    waveform, sr = sf.read(item_path)
                    feat_item = mfcc(waveform+1e-9, maxfreq=sr/2.0, nwin=.128, shift=.032)[0]
                    feat_delta1 = compute_delta(feat_item)
                    feat_delta2 = compute_delta(feat_delta1)
                    feat = np.concatenate((feat_item, feat_delta1, feat_delta2), axis=1)
            else:
                if self.precomputed:
                    feat = np.load(item_path)[:,:,0].T
                else:
                    waveform, sr = sf.read(item_path)
                    feat_item = plp(waveform+1e-9, fs=sr, rasta=False, nwin=.128, shift=.032)[0]
                    feat_delta1 = compute_delta(feat_item)
                    feat_delta2 = compute_delta(feat_delta1)
                    feat = np.concatenate((feat_item, feat_delta1, feat_delta2), axis=1)



            mirror_feat = np.pad(feat, ((10,), (0,)), 'reflect')
            frames = []
            for j in range(10 + i*25, 10 + (i+1)*25):
                frames.append(np.reshape(mirror_feat[j-10:j+11,:], -1))
            X[25*i:25*(i+1),] = np.array(frames)

            # Store class
            label = item_path.split('/')[-3]

            target_i = keras.utils.to_categorical(self.target_to_class[label], num_classes=self.n_classes)
            target[25*i:25*(i+1),:] = np.repeat(target_i.reshape((1,-1)),repeats=25, axis=0)
        return X, target

    def __data_generation_lstm(self, item_path):
        # Generate data
        waveform, sr = sf.read(item_path)
        if self.feat=='mfcc':
            feat_item = mfcc(waveform+1e-9, maxfreq=sr/2.0)[0]
        else:
            feat_item = plp(waveform+1e-9, fs=sr, rasta=False)[0]
        feat_delta1 = compute_delta(feat_item)
        feat_delta2 = compute_delta(feat_delta1)
        feat = np.concatenate((feat_item, feat_delta1, feat_delta2), axis=1)
        # Store class
        label = item_path.split('/')[-3]
        y = self.target_to_class[label]
        target = np.zeros((feat.shape[0], self.n_classes))
        target[:,y] = 1
        return feat.reshape(1,*feat.shape), target

    
class DataTfLoader:
    
    def __init__(self, root_path):
        
        self.root_path = root_path
        targets = os.listdir(self.root_path)
        self.list_dir = []
        self.dim = (128, 79)
        self.target_to_class = {lang: i for i, lang in enumerate(targets)}
        self.n_classes = len(self.target_to_class)
        i = 0
        for target in targets:
            target_path = os.path.join(os.path.join(self.root_path, target), 'clips_cut/')
            for path in os.listdir(target_path):
                if (not path.startswith('.')):
                    item_path = os.path.join(target_path, path)
                    self.list_dir.append(item_path)
                
            i += 1
            
        self.data_loader = tf.data.Dataset.list_files(self.list_dir)
    
    def getLoader(self):
        
        
        @tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
        def tf_function(input):
            value = tf.py_function(extract_spectogram, [input], (tf.float32, tf.float32))
            value[0].set_shape([128, 79, 1])
            value[1].set_shape([None])
            return value


        def extract_spectogram(item_path):
            item_path = item_path.numpy().decode('utf-8')
            waveform, sr = sf.read(item_path)
            mel = librosa.feature.melspectrogram(waveform, sr=16000)
            ps_db = librosa.power_to_db(mel, ref=np.max).reshape((*self.dim, 1))
            X = (ps_db - np.mean(ps_db))/np.var(ps_db)
            # Store class
            label = item_path.split('/')[-3]
            y = self.target_to_class[label]

            target = keras.utils.to_categorical(y, num_classes=self.n_classes)
            return X, target
        
        return self.data_loader.map(tf_function, num_parallel_calls=24)
"""

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

"""
