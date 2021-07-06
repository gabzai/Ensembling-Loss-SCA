import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import keras.backend as K
import random
import gc


from keras.models import Model
from keras.layers import Activation, Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization, MaxPooling1D, concatenate
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn import preprocessing

def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return

class Logger(object):
    def __init__(self, name_file):
        self.terminal = sys.stdout
        self.log = open(str(name_file)+".log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

# Naive implementation
def ensembling_sca(score_vector, nb_models=5, nb_class=3, alpha_value=0.1, beta_value=0.1, gamma_value=0.1, mu_value=1.0):

    # Ensembling loss function
    def ensembling_loss_sca(y_true, y_pred):

        alpha = K.constant(alpha_value, dtype='float32')
        beta = K.constant(beta_value, dtype='float32')
        gamma = K.constant(gamma_value, dtype='float32')
        mu = K.constant(mu_value, dtype='float32')
        cst_epsi = K.constant(0.1, dtype='float32')

        # Batch_size initialization
        y_true_int = K.cast(y_true, dtype='int32')
        batch_s = K.cast(K.shape(y_true_int)[0],dtype='int32')

        # Indexing the training set (shape(range_value) = (?,))
        range_value = K.arange(0, batch_s, dtype='int64')

        # Relevance Loss Initialization
        rel_loss = 0
    
        for i in range(nb_models):

            # Get rank and scores associated with the secret key (shape(rank_sk) = (?,))
            values_topk_logits, indices_topk_logits = tf.nn.top_k(score_vector[i], k=nb_class, sorted=True) # shape(values_topk_logits) = (?, nb_class) ; shape(indices_topk_logits) = (?, nb_class)
            rank_sk = tf.where(tf.equal(K.cast(indices_topk_logits, dtype='int64'), tf.reshape(K.argmax(y_true_int), [tf.shape(K.argmax(y_true_int))[0], 1])))[:,1] + 1 # Index of the correct output among all the hypotheses (shape = (?,))
            s_sk = tf.gather_nd(values_topk_logits, K.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), tf.reshape(rank_sk-1, [tf.shape(values_topk_logits)[0], 1])])) # Score of the secret key (shape = (?,))

            for j in range(nb_class):

                # Score for each key hypothesis (shape(s_j) = (?,))
                s_j = tf.gather_nd(values_topk_logits, K.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), j*tf.ones([tf.shape(values_topk_logits)[0], 1], dtype='int64')]))

                # Indicator function identifying when (j == secret key)
                indicatrice_approximation = tf.ones(batch_s) - K.cast(K.equal(rank_sk-1, j), dtype='float32') * tf.ones(batch_s)

                # Logistic loss computation
                logistic_loss = K.cast(K.log(1 + K.exp(- alpha * (s_sk - s_j)))/K.log(2.0), dtype='float32')

                # Relevance Loss computation
                rel_loss = K.cast(tf.reduce_sum((indicatrice_approximation * logistic_loss))+rel_loss, dtype='float32')


        # Redundancy Loss Initialization
        red_loss = 0

        for i in range(nb_models-1):
            for j in range(i+1, nb_models):
                for k in range(nb_class):
                    for l in range(nb_class):
                        # Score related to each key hypothesis for the committee member Fi (shape(s_i) = (?,)) and Fj (shape(s_j) = (?,))  
                        s_i = tf.gather_nd(score_vector[i], K.concatenate([tf.reshape(range_value, [tf.shape(score_vector[i])[0], 1]), k*tf.ones([tf.shape(score_vector[i])[0], 1], dtype='int64')]))
                        s_j = tf.gather_nd(score_vector[j], K.concatenate([tf.reshape(range_value, [tf.shape(score_vector[j])[0], 1]), l*tf.ones([tf.shape(score_vector[j])[0], 1], dtype='int64')]))

                        # Logistic loss computation
                        logistic_loss = K.cast(K.log(1 - K.exp(- cst_epsi - (gamma * (K.abs(s_i - s_j)))))/K.log(2.0), dtype='float32')

                        # Redundancy Loss computation
                        red_loss = K.cast(tf.reduce_sum(logistic_loss)+red_loss, dtype='float32')

        # Conditional Redundancy Loss Initialization
        cond_red_loss = 0

        for i in range(nb_models-1):

            # Get rank and scores associated with the secret key (shape(rank_sk) = (?,)) for the committee member Fi
            values_topk_logits_i, indices_topk_logits_i = tf.nn.top_k(score_vector[i], k=nb_class, sorted=True) # shape(values_topk_logits) = (?, nb_class) ; shape(indices_topk_logits) = (?, nb_class)
            rank_sk_i = tf.where(tf.equal(K.cast(indices_topk_logits_i, dtype='int64'), tf.reshape(K.argmax(y_true_int), [tf.shape(K.argmax(y_true_int))[0], 1])))[:,1] + 1 # Index of the correct output among all the hypotheses (shape = (?,))
            s_sk_i = tf.gather_nd(values_topk_logits_i, K.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits_i)[0], 1]), tf.reshape(rank_sk_i-1, [tf.shape(values_topk_logits_i)[0], 1])])) # Score of the secret key (shape = (?,))

            for j in range(i+1, nb_models):
                
                # Get rank and scores associated with the secret key (shape(rank_sk) = (?,)) for the committee member Fj
                values_topk_logits_j, indices_topk_logits_j = tf.nn.top_k(score_vector[j], k=nb_class, sorted=True) # shape(values_topk_logits) = (?, nb_class) ; shape(indices_topk_logits) = (?, nb_class)
                rank_sk_j = tf.where(tf.equal(K.cast(indices_topk_logits_j, dtype='int64'), tf.reshape(K.argmax(y_true_int), [tf.shape(K.argmax(y_true_int))[0], 1])))[:,1] + 1 # Index of the correct output among all the hypotheses (shape = (?,))
                s_sk_j = tf.gather_nd(values_topk_logits_j, K.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits_j)[0], 1]), tf.reshape(rank_sk_j-1, [tf.shape(values_topk_logits_j)[0], 1])])) # Score of the secret key (shape = (?,))

                # Logistic loss computation
                logistic_loss = K.cast(K.log(K.exp(- beta * (K.abs(s_sk_i - s_sk_j))))/K.log(2.0), dtype='float32')

                # Conditional Redundancy Loss computation
                cond_red_loss = K.cast(tf.reduce_sum(logistic_loss)+cond_red_loss, dtype='float32')

        reg_factor = 2*mu/(nb_models * (nb_models-1))
        denom_rel = nb_models

        return (rel_loss/(K.cast(batch_s, dtype='float32')*denom_rel))-reg_factor*((red_loss/K.cast(batch_s, dtype='float32'))+(cond_red_loss/K.cast(batch_s, dtype='float32')))

    return ensembling_loss_sca

# Acc of the ensemble model
def acc_metric(predictions):

    def acc_ensembling(y_true, y_pred):

        ens_p = tf.reduce_mean(predictions, axis=0)
        return categorical_accuracy(y_true, ens_p)

    return acc_ensembling

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)


### CNN Model Random (Model i)
def cnn_architecture_random(input_size=13000, classes=3, nb_layers=1, nb_filters=2, length_filter=1, pooling_operator="average_pooling", pooling_stride=2, nb_fc_layers=0, nb_nodes_fc=2):

    # Personal design
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape, dtype='float32')

    # Convolutional Part
    for i in range(nb_layers):
        if (i==0):
            x = Conv1D(nb_filters[i], length_filter[i], kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
        else:
            x = Conv1D(nb_filters[i], length_filter[i], kernel_initializer='he_uniform', activation='selu', padding='same')(x)
        x = BatchNormalization()(x)
        if (pooling_operator[i] == "average_pooling"):
            x = AveragePooling1D(pooling_stride[i], strides=pooling_stride[i])(x)
        else:
            x = MaxPooling1D(pooling_stride[i], strides=pooling_stride[i])(x)

    # Flatten Layer
    x = Flatten()(x)

    if (nb_fc_layers!=0):
        for j in range(nb_fc_layers):
            x = Dense(nb_nodes_fc[j], kernel_initializer='he_uniform', activation='selu')(x)

    # Logits Layer
    score_layer = Dense(classes, activation=None)(x)
    predictions = Activation('softmax')(score_layer)

    # Create Model
    inputs = img_input
    model = Model(inputs, predictions)

    return model, inputs, score_layer, predictions

### Ensemble Model
def ensemble_model(model_dic, nb_models=5, classes=3, learning_rate=0.001, alpha_value=1.0, beta_value=1.0, gamma_value=1.0, mu_value=1.0):
    models = []
    inputs = []
    scores = []
    predictions = []
    for i in range(nb_models):
        models.append(model_dic[str(i)][0])
        inputs.append(model_dic[str(i)][1])
        scores.append(model_dic[str(i)][2])
        predictions.append(model_dic[str(i)][3])

    if (nb_models>1):
        model_output = concatenate(predictions)
    else:
        model_output = predictions

    # Create model
    model = Model(input=inputs, output=model_output)
    optimizer = Adam(lr=learning_rate)

    model.compile(loss=ensembling_sca(scores, nb_models=nb_models, nb_class=classes, alpha_value=alpha_value, beta_value=beta_value, gamma_value=gamma_value, mu_value=mu_value), optimizer=optimizer, metrics=[acc_metric(predictions)])

    return model


#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100, max_lr=1e-3, nb_models=5, nb_classes=3):
    check_file_exists(os.path.dirname(save_file_name))
    
    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name, monitor='val_acc_ensembling', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check
    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)

    Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    X_profiling_inputs=[]
    X_test_inputs=[]
    for i in range(nb_models):
        X_profiling_inputs.append(Reshaped_X_profiling)
        X_test_inputs.append(Reshaped_X_test)
    
    callbacks=[save_model]

    history = model.fit(x=X_profiling_inputs, y=to_categorical(Y_profiling, num_classes=nb_classes), validation_data=(X_test_inputs, to_categorical(Y_test, num_classes=nb_classes)), batch_size=batch_size, verbose = 1, epochs=epochs, callbacks=callbacks)
    return history


#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

root = "./"
trained_models_folder = root+"trained_models/"
history_folder = root+"training_history/"

# Import file
filename_trace = "" #Define the filename including the leakage traces 
filename_labels = "" #Define the filename including the labels 

# Hyperparameters Selection
nb_models = 32
input_size = 13000
nb_classes = 3

nb_epochs = 100
batch_size = 128
learning_rate = 1e-3

# Individual Loss Parameter
alpha = 0.1
gamma = 0.1
beta = 0.1

mu = 1 # Regularization term impacting the diversity of the ensemble model

train_data_len = 30000
valid_data_len = 3000

model_name = "Ensemble_Model"
    
# Log Initialization
sys.stdout = Logger(model_name)

print('Ensemble Model !')

print('\n############### Configuration ###############\n')
print('nb_models = ', nb_models)
print('input_size = ', input_size)
print('nb_classes = ', nb_classes)

print('nb_epochs = ', nb_epochs)
print('batch_size = ', batch_size)
print('learning_rate = ', learning_rate)


print('alpha = ', alpha)
print('gamma = ', gamma)
print('beta = ', beta)
print('regularization term =', mu)

print('input_size = ', input_size)
print('train_data_len = ', train_data_len)
print('valid_data_len = ', valid_data_len)

start = time.time()

nb_traces = train_data_len + valid_data_len

print('\n############### Chargement des traces ###############\n')
X = np.load(root+'dataset/%s.npy'%filename_trace)[:nb_traces]
Y = np.load(root+'dataset/%s.npy'%filename_labels)[:nb_traces]

print('Traces and Labels loaded ! \n')

# Shuffle data
(X, Y) = shuffle_data(X, Y)

X_profiling, X_valid = X[:train_data_len], X[train_data_len:train_data_len+valid_data_len]
Y_profiling, Y_valid = Y[:train_data_len], Y[train_data_len:train_data_len+valid_data_len]


print('\n############### Preprocessing ###############\n')
#Standardization (0 mean and unit variance)
scaler = preprocessing.StandardScaler()
X_profiling = scaler.fit_transform(X_profiling)
X_valid = scaler.transform(X_valid)

X_profiling = X_profiling.astype('float32')
X_valid = X_valid.astype('float32')


#################################################
#################################################

####                  Train                ######

#################################################
#################################################

print("\n############### Starting Training #################\n")

# Choose the range of hyperparameters
print('\n Model name = '+model_name)
model_dic = {}
for i in range(nb_models):
    nb_layers_set = random.randint(1,2)
    nb_filters_set = []
    length_filter_set = []
    pooling_operator_set = []
    pooling_stride_set = []
    for j in range(nb_layers_set):
        nb_filters_set.append(random.choice([2, 4, 8]))
        length_filter_set.append(random.choice([1, 5, 11, 21]))
        pooling_operator_set.append(random.choice(['average_pooling', 'max_pooling']))
        pooling_stride_set.append(random.choice([2, 4, 6]))

    nb_fc_layers_set = random.randint(0,2)
    nb_nodes_fc_set = []
    if (nb_fc_layers_set!=0):
        for j in range(nb_fc_layers_set):
            nb_nodes_fc_set.append(random.choice([2, 4, 8]))

    model_dic[str(i)] = cnn_architecture_random(input_size=input_size, classes=nb_classes, nb_layers=nb_layers_set, nb_filters=nb_filters_set, length_filter=length_filter_set, pooling_operator=pooling_operator_set, pooling_stride=pooling_stride_set, nb_fc_layers=nb_fc_layers_set, nb_nodes_fc=nb_nodes_fc_set)

# Construction of the Ensemble Model
model_ensemble = ensemble_model(model_dic, nb_models=nb_models, classes=nb_classes, learning_rate=learning_rate, alpha_value=alpha, beta_value=beta, gamma_value=gamma, mu_value=mu)
print(model_ensemble.summary())

# Record the metrics
history = train_model(X_profiling, Y_profiling, X_valid, Y_valid, model_ensemble, trained_models_folder + model_name + "_{epoch:02d}-{val_acc_ensembling:.4f}", epochs=nb_epochs, batch_size=batch_size, max_lr=learning_rate, nb_models=nb_models, nb_classes=nb_classes)

end=time.time()
print('Temps execution = %d'%(end-start))

print("\n############### Training Done #################\n")

# Save the metrics
with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

print('Training Process Done !')

K.clear_session()
gc.collect()
