import numpy
import scipy.signal
import os
import os.path
import scipy.io.wavfile
import parameters
import data_entry
import sys
import train_data

def set_signal_length(signal, length_sec):
    signal_length_samples = int(length_sec*parameters.SAMPLE_RATE_HZ)

    if(len(signal) >= signal_length_samples):
        return signal[0:signal_length_samples]

    # if signal is shorter, zero pad
    size_of_padding = signal_length_samples-len(signal)
    padded_signal = numpy.append(signal, [0]*size_of_padding)
    return padded_signal


def load_wav_file(file_path):
    (fs, wav_data) = scipy.io.wavfile.read(file_path)
    assert (fs == parameters.SAMPLE_RATE_HZ)
    assert isinstance(wav_data, numpy.ndarray)

    wav_data = wav_data / (float(2 ** 16))  # Convert audio array to float
    wav_data = set_signal_length(wav_data, parameters.SIGNAL_LENGTH_SEC)
    return wav_data

def getSourceSignals(file_list, orig_dir):
    signal_list = []
    while(len(signal_list) < parameters.NUM_OF_SOURCES_IN_SIGNAL):
        if(len(file_list) == 0):
            raise Exception('Not enough input wav files in folder')

        file_name = file_list.pop(0)

        ext = str(file_name.split(r'.')[-1])
        if (ext.lower() != 'wav'):
            continue

        file_path = os.path.join(orig_dir, file_name)
        wav_data = load_wav_file(file_path)
        signal_list.append(wav_data)

    return signal_list

def build_train_dataset():
    list_dir = os.listdir(parameters.TRAIN_SENTENCES_FOLDER)

    train_data_inst = train_data.TrainData()
    for ind in range(parameters.NUM_OF_TRAIN_SIGNALS):
        signal_list = getSourceSignals(list_dir, parameters.TRAIN_SENTENCES_FOLDER)
        record_folder_path = os.path.join(parameters.OUTPUT_FOLDER,
                                          r'Train\Data_Set_{0}_Records'.format(ind+1))
        cur_data_entry = data_entry.DataEntry(signal_list, parameters.BRIR_FILE, record_folder_path)
        train_data_inst.addDataEntry(cur_data_entry)

        # Save data-set record
        cur_data_entry.saveDataSetRecord()

        # Print progress
        progress = int(100 * float(ind+1) / parameters.NUM_OF_TRAIN_SIGNALS)
        sys.stdout.write('Creating training data-set: {0:3d}% \r'.format(progress))
        sys.stdout.flush()

    assert (train_data_inst.isDataFull() == True)
    train_data_inst.meanVarianceNormalize()

    sys.stdout.write('\n')
    sys.stdout.flush()
    print('Done creating data-set')
    return train_data_inst


def build_test_dataset(mean, std):
    list_dir = os.listdir(parameters.TEST_SENTENCES_FOLDER)

    data_entries = []
    while (len(data_entries) < parameters.NUM_OF_TEST_SIGNALS):
        signal_list = getSourceSignals(list_dir, parameters.TEST_SENTENCES_FOLDER)
        record_folder_path = os.path.join(parameters.OUTPUT_FOLDER,
                                          r'Test\Data_Set_{0}_Records'.format(len(data_entries)))
        cur_data_entry = data_entry.DataEntry(signal_list, parameters.BRIR_FILE, record_folder_path)
        cur_data_entry.meanVarianceNormalize(mean, std)
        data_entries.append(cur_data_entry)

        # Save data-set record
        cur_data_entry.saveDataSetRecord()

        # Print progress
        progress = int(100 * float(len(data_entries)) / parameters.NUM_OF_TEST_SIGNALS)
        sys.stdout.write('Creating test data-set: {0:3d}% \r'.format(progress))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()
    print('Done creating data-set')
    return data_entries

if __name__ == '__main__':
    build_train_dataset()
