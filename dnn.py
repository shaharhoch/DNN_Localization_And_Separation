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

def initNet(in_dim, out_dim):
    assert len(out_dim) == 2

    #Create input layer
    inputs = Input(shape = (in_dim,))

    #Create hidden layers
    temp_layer = inputs
    for layer_size in parameters.DNN_HIDDEN_LAYER_SIZE:
        temp_layer = Dense(output_dim=layer_size, activation='sigmoid')(temp_layer)

    #Create output
    output_layers = []
    for ind in range(out_dim[0]):
        out_layer = Dense(output_dim=out_dim[1], activation='softmax',name='out_{0}'.format(ind))(temp_layer)
        output_layers.append(out_layer)

    #Inlude all layers in a model
    model = Model(input=inputs, output=output_layers)

    #Use default values for adaptive gradient decent optimizer, as recommended in Keras documentation
    optimizer = Adagrad()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

def dataEntriesToArray(data_entries):
    data_in = numpy.array([])

    #Init data target
    data_target = []
    for ind in range(parameters.SGRAM_NUM_CHANNELS):
        data_target.append(numpy.array([]))

    for entry in data_entries:
        assert isinstance(entry, DataEntry)
        data_in = create_mixtures.generalizedConcat((data_in, entry.features), dim = 0)

        for ind in range(len(data_target)):
            data_target[ind] = create_mixtures.generalizedConcat((data_target[ind], entry.targets[ind]), dim=0)

    return (data_in, data_target)

def plotTrainAccuracy(history):
    assert isinstance(history, callbacks.History)

    # Calculate total average train and validation accuracy
    val_acc = numpy.zeros(parameters.MAX_EPOCHS_TRAIN)
    train_acc = numpy.zeros(parameters.MAX_EPOCHS_TRAIN)
    for ind in range(parameters.SGRAM_NUM_CHANNELS):
        val_acc = val_acc + numpy.array(history.history['val_out_{0}_categorical_accuracy'.format(ind)])
        train_acc = train_acc + numpy.array(history.history['out_{0}_categorical_accuracy'.format(ind)])

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
    data_entries = create_mixtures.build_train_dataset()
    (train_input, train_target) = dataEntriesToArray(data_entries)

    net = initNet(train_input.shape[1], [parameters.SGRAM_NUM_CHANNELS, parameters.NUM_OF_DIRECTIONS+1])

    history = net.fit(train_input, train_target, batch_size=100, nb_epoch=parameters.MAX_EPOCHS_TRAIN, validation_split=0.15)
    plotTrainAccuracy(history)

    test_entries = create_mixtures.build_test_dataset()
    avg_source_fa = 0
    avg_source_md = 0
    for entry in test_entries:
        assert isinstance(entry, DataEntry)
        performance = entry.estimateNetPerformance(net)
        avg_source_fa = avg_source_fa+performance['source_fa']
        avg_source_md = avg_source_md+performance['source_md']

    avg_source_fa = avg_source_fa/len(test_entries)
    avg_source_md = avg_source_md / len(test_entries)

    print('Average Source FA: {0}%'.format(avg_source_fa*100))
    print('Average Source MD: {0}%'.format(avg_source_md*100))

