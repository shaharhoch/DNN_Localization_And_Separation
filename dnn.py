import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Dense, Input, Merge
from keras.models import Model, Sequential
from keras.optimizers import Adagrad
from keras import callbacks
import parameters
import create_mixtures
import numpy
from data_entry import DataEntry
import matplotlib.pyplot as plt
import pickle

TRAIN_DATA_FILE = 'train_data.pkl'

def initNet(in_dim, out_dim):
    #Create input layer
    inputs = Input(shape = (in_dim,))

    #Create hidden layers
    temp_layer = inputs
    for layer_size in parameters.DNN_HIDDEN_LAYER_SIZE:
        temp_layer = Dense(output_dim=layer_size, activation='sigmoid')(temp_layer)

    #Create output
    output_layer = Dense(output_dim=out_dim, activation='softmax', name='out')(temp_layer)

    #Inlude all layers in a model
    model = Model(input=inputs, output=output_layer)

    #Use default values for adaptive gradient decent optimizer, as recommended in Keras documentation
    optimizer = Adagrad()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

def dataEntriesToArray(data_entries):
    #Init data input and target
    data_in = []
    data_target = []
    for ind in range(parameters.SGRAM_NUM_CHANNELS):
        data_target.append(numpy.array([]))
        data_in.append(numpy.array([]))

    for entry in data_entries:
        assert isinstance(entry, DataEntry)
        for ind in range(len(data_target)):
            data_target[ind] = DataEntry.generalizedConcat((data_target[ind], entry.targets[ind]), dim=0)
            data_in[ind] = DataEntry.generalizedConcat((data_in[ind], entry.features[ind]), dim=0)

    return (data_in, data_target)

def plotTrainAccuracy(nets_history):
    # Calculate total average train and validation accuracy
    val_acc = numpy.zeros(parameters.MAX_EPOCHS_TRAIN)
    train_acc = numpy.zeros(parameters.MAX_EPOCHS_TRAIN)
    for history in nets_history:
        assert isinstance(history, callbacks.History)
        val_acc = val_acc + numpy.array(history.history['val_categorical_accuracy'])
        train_acc = train_acc + numpy.array(history.history['categorical_accuracy'])

    val_acc = val_acc/parameters.SGRAM_NUM_CHANNELS
    train_acc = train_acc/parameters.SGRAM_NUM_CHANNELS

    #Plot accuracy
    fig = plt.figure()
    plt.plot(val_acc)
    plt.plot(train_acc)
    plt.title('Train and Validation categorical accuracy during training')
    plt.ylabel('Categorical Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['validation', 'training'], loc='upper left')
    save_path = os.path.join(parameters.OUTPUT_FOLDER, 'Net_Accuracy')
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    training_data_file = os.path.join(parameters.OUTPUT_FOLDER, TRAIN_DATA_FILE)
    if(os.path.exists(training_data_file)):
        print('Found training data file, loading...')
        file_read = open(training_data_file, 'rb')
        (train_input, train_target) = pickle.load(file_read)
        file_read.close()
        print('Training data loaded.')
    else:
        print('Training data file was not found.')
        data_entries = create_mixtures.build_train_dataset()
        (train_input, train_target) = dataEntriesToArray(data_entries)

        print('Saving training data file...')
        file_write = open(training_data_file, 'wb')
        pickle.dump((train_input, train_target), file_write)
        file_write.close()
        print('Training data file saved.')

    # Train all the needed DNNs
    nets = []
    nets_history = []
    for ind in range(parameters.SGRAM_NUM_CHANNELS):
        print('Training net {0} out of {1}:'.format(ind+1, parameters.SGRAM_NUM_CHANNELS))
        net = initNet(train_input[ind].shape[1], train_target[ind].shape[1])
        nets.append(net)
        history = net.fit(train_input[ind], train_target[ind], batch_size=100, nb_epoch=parameters.MAX_EPOCHS_TRAIN,
                          validation_split=0.15)
        nets_history.append(history)

    plotTrainAccuracy(nets_history)

    test_entries = create_mixtures.build_test_dataset()
    avg_source_fa = 0
    avg_source_md = 0
    for entry in test_entries:
        assert isinstance(entry, DataEntry)
        performance = entry.estimateNetsPerformance(nets)
        avg_source_fa = avg_source_fa+performance['source_fa']
        avg_source_md = avg_source_md+performance['source_md']

    avg_source_fa = avg_source_fa/len(test_entries)
    avg_source_md = avg_source_md / len(test_entries)

    print('Average Source FA: {0}%'.format(avg_source_fa*100))
    print('Average Source MD: {0}%'.format(avg_source_md*100))

