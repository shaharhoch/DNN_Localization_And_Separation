import numpy
import scipy.signal
import os
import os.path
import scipy.io.wavfile
import parameters
import data_entry
import sys

BRIR_FILE = r'..\Database\BRIR\UniS_Anechoic_BRIR_16k.sofa'
SENTENCES_FOLDER = r'..\Database\Clean\Train_Data'

def set_signal_length(signal, length_sec):
    signal_length_samples = int(length_sec*parameters.SAMPLE_RATE_HZ)

    if(len(signal) >= signal_length_samples):
        return signal[0:signal_length_samples]

    #if signal is shorter, zero pad
    size_of_padding = signal_length_samples-len(signal)
    padded_signal = numpy.append(signal, [0]*size_of_padding)
    return padded_signal

def load_wav_file(file_path):
    (fs, wav_data) = scipy.io.wavfile.read(file_path)
    assert (fs == parameters.SAMPLE_RATE_HZ)
    wav_data = wav_data / (float(2 ** 16))  # Convert audio array to float
    wav_data = set_signal_length(wav_data, parameters.SIGNAL_LENGTH_SEC)
    return wav_data

def getSourceSignals(file_list, orig_dir):
    signal_list = []
    while(len(signal_list)<parameters.NUM_OF_SOURCES_IN_SIGNAL):
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

def generalizedConcat(in_data, dim):
    list_in_data = list(in_data)
    out_stacked = []

    for data in list_in_data:
        if(len(data) == 0):
            continue
        if(len(out_stacked) == 0):
            out_stacked = data
            continue

        out_stacked = numpy.concatenate((out_stacked, data), axis=dim)

    return out_stacked

def build_dataset():
    list_dir = os.listdir(SENTENCES_FOLDER)

    data_entries = []
    while(len(data_entries) < parameters.NUM_OF_SIGNALS):
        signal_list = getSourceSignals(list_dir, SENTENCES_FOLDER)
        cur_data_entry = data_entry.DataEntry(signal_list, BRIR_FILE)
        data_entries.append(cur_data_entry)

        #Save data-set record
        record_folder_path = os.path.join(parameters.OUTPUT_FOLDER, 'Data_Set_{0}_Records'.format(len(data_entries)))
        cur_data_entry.saveDataSetRecord(record_folder_path)

        # Print progress
        progress = int(100 * float(len(data_entries)) / parameters.NUM_OF_SIGNALS)
        sys.stdout.write('Creating data-set: {0:3d}% \r'.format(progress))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()
    print('Done creating data-set')
    return data_entries

if __name__ == '__main__':
    build_dataset()