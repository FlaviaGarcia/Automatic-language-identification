from models.DNN import DNN
from models import utils
from tensorflow.keras.utils import to_categorical


## Add context and prepare data so DNN accepts it 

## Create fake data to test that everything is working 
n_frames_utterance = 10 
fake_features, fake_targets = utils.generate_fake_data(n_utterances=10,n_frames_utterance=n_frames_utterance)
fake_targets_categ = to_categorical(fake_targets)

x_train = fake_features[:70]
y_train = fake_targets_categ[:70]
x_val = fake_features[70:80]
y_val = fake_targets_categ[70:80]
x_test = fake_features[80:]
y_test = fake_targets_categ[80:]

n_input_nodes=fake_features.shape[1]
n_output_nodes=fake_targets_categ.shape[1]

n_hidden_nodes = [2560]
 
batch_normalization=False
dropout=False
batch_size=10 
n_epochs=20 

dnn = DNN(n_input_nodes, n_hidden_nodes, n_output_nodes, 
          batch_normalization, dropout)

dnn.train(x_train, y_train, x_val, y_val, batch_size, n_epochs)

scores = dnn.get_scores(x_test, n_frames_utterance)

classes = dnn.get_classes(x_test, n_frames_utterance)