import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adagrad
from keras import callbacks
import parameters
import create_mixtures
import numpy
from data_entry import DataEntry
import matplotlib.pyplot as plt
import dill
from train_data import TrainData

TRAIN_DATA_FILE = 'train_data.dill'
NET_SAVE_FILE = 'net_save.hdf5'

def initNet(in_dim, out_dim):
    assert len(out_dim) == 2

    print('Initializing DNN...')
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

    print('DNN Initialized.')
    return model

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

def estimateTestPerformance(train_data, net):
    assert isinstance(train_data, TrainData)

    test_entries = create_mixtures.build_test_dataset(train_data.mean, train_data.std)
    avg_source_fa = 0
    avg_source_md = 0
    avg_pesq = 0
    for entry in test_entries:
        assert isinstance(entry, DataEntry)
        performance = entry.estimateNetPerformance(net)
        avg_source_fa = avg_source_fa + performance['source_fa']
        avg_source_md = avg_source_md + performance['source_md']
        avg_pesq = avg_pesq + performance['PESQ']

    avg_source_fa = avg_source_fa / len(test_entries)
    avg_source_md = avg_source_md / len(test_entries)
    avg_pesq = avg_pesq/ len(test_entries)

    print('Average Source FA: {0}%'.format(avg_source_fa * 100))
    print('Average Source MD: {0}%'.format(avg_source_md * 100))
    print('Average PESQ: {0}'.format(avg_pesq))

def getTrainingData():
    training_data_file = os.path.join(parameters.OUTPUT_FOLDER, TRAIN_DATA_FILE)
    if (os.path.exists(training_data_file)):
        print('Found training data file, loading...')
        file_read = open(training_data_file, 'rb')
        train_data = dill.load(file_read)
        file_read.close()
        print('Training data loaded.')
    else:
        print('Training data file was not found.')
        train_data = create_mixtures.build_train_dataset()

        print('Saving training data file...')
        file_write = open(training_data_file, 'wb')
        dill.dump(train_data, file_write)
        print('Training data file saved.')

    assert isinstance(train_data, TrainData)
    return train_data

def saveNet(net):
    assert isinstance(net, Model)

    print('Saving DNN...')

    net_file_path = os.path.join(parameters.OUTPUT_FOLDER, NET_SAVE_FILE)
    net.save(net_file_path, overwrite=True)

    print('DNN saved.')

def main():
    train_data = getTrainingData()
    assert isinstance(train_data, TrainData)

    net = initNet(parameters.getSizeOfFeatureVec(), [parameters.SGRAM_NUM_CHANNELS, parameters.NUM_OF_DIRECTIONS + 1])
    history = net.fit(train_data.getTrainInputs(), train_data.getTrainTargets(), batch_size=100,
                      nb_epoch=parameters.MAX_EPOCHS_TRAIN, validation_split=0.15, verbose=2)
    saveNet(net)

    plotTrainAccuracy(history)
    estimateTestPerformance(train_data, net)

if __name__ == '__main__':
    out_folder_save = parameters.OUTPUT_FOLDER

    print('Run number 1, STFT, only binaural')
    parameters.OUTPUT_FOLDER = out_folder_save + '_RUN_1'
    parameters.SGRAM_TYPE = 'STFT'
    parameters.USE_MONAURAL_FEATURES = False
    main()

    print('Run number 2, STFT, monaural and binaural')
    parameters.OUTPUT_FOLDER = out_folder_save + '_RUN_2'
    parameters.SGRAM_TYPE = 'STFT'
    parameters.USE_MONAURAL_FEATURES = True
    main()

    print('Run number 3, Cgram, only binaural')
    parameters.OUTPUT_FOLDER = out_folder_save + '_RUN_3'
    parameters.SGRAM_TYPE = 'CGRAM'
    parameters.USE_MONAURAL_FEATURES = False
    main()

    print('Run number 4, CGRAM, monaural and binaural')
    parameters.OUTPUT_FOLDER = out_folder_save + '_RUN_4'
    parameters.SGRAM_TYPE = 'CGRAM'
    parameters.USE_MONAURAL_FEATURES = True
    main()
