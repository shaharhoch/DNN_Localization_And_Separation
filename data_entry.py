import numpy
import h5py
import random
import scipy.signal
import parameters
import features
import os.path
import matplotlib.pyplot as plt
from keras.models import Model
import scipy.io.wavfile

class DataEntry():

    '''
    Lets assume we have K signals in the data set
    Input:
        in_signals- a K length list, each element is an input signal
    Output:
        signals- a K length list, each element is a 2-D array which represents a directional sound
        angles- a K length list which contain the source angles of signals
        targets- the training targets for the data set
        res_signal - the sound signal containing the signal combination
    '''
    def __init__(self, in_signals, brir_file_path, save_folder):
        brir_file = h5py.File(brir_file_path)
        brir = numpy.array(brir_file.get('Data.IR'))
        pos_def = numpy.array(brir_file.get('SourcePosition'))

        self.save_folder = save_folder
        self.signals = []
        self.angles = []
        self.noise = numpy.zeros(in_signals[0].shape)

        signal_length_samples = int(parameters.SIGNAL_LENGTH_SEC*parameters.SAMPLE_RATE_HZ)
        self.res_signal = numpy.zeros((signal_length_samples, 2))
        for signal in in_signals:
            assert (len(signal) == signal_length_samples)
            # Normalize signal power
            sig_rms = self.getArrayRms(signal)
            signal = (signal/sig_rms)*parameters.INPUT_SIGNAL_RMS

            self.angles.append(DataEntry.getRandomAngle(self.angles))
            binaural_signal = DataEntry.getBinauralSound(signal, brir, pos_def, self.angles[-1])
            self.signals.append(binaural_signal)

            self.res_signal = self.res_signal+binaural_signal

        self.updateTargetsFromSignals()
        self.updateFeaturesFromResSignal()

    @classmethod
    def getArrayRms(cls, arr_in):
        assert isinstance(arr_in, numpy.ndarray)
        assert (len(arr_in.shape) == 1)

        rms = numpy.sqrt(numpy.linalg.norm(arr_in)/arr_in.size)
        return rms

    def updateTargetsFromSignals(self):
        cgrams = []
        for binaural_signal in self.signals:
            cgram = features.getCochleagram(binaural_signal[:,0])
            cgrams.append(cgram)

        noise_cgram = features.getCochleagram(self.noise)

        #Init target with zeros
        targets = []
        for ind in range(cgrams[0].shape[1]):
            targets.append(numpy.zeros((cgrams[0].shape[0], parameters.NUM_OF_DIRECTIONS+1)))

        for out_ind in range(cgrams[0].shape[1]):
            for time_ind in range(cgrams[0].shape[0]):
                cgram_vals = numpy.array([])
                for cgram in cgrams:
                    cgram_vals = numpy.append(cgram_vals, cgram[time_ind,out_ind])

                max_ind = cgram_vals.argmax()
                max_val = cgram_vals.max()
                max_angle = self.angles[max_ind]
                angle_ind = DataEntry.getAngleIdx(max_angle)

                noise_th = noise_cgram[time_ind, out_ind]*(10**(parameters.CGRAM_NOISE_TH_dB/10))
                if(max_val < noise_th):
                    targets[out_ind][time_ind,-1] = 1
                else:
                    targets[out_ind][time_ind, angle_ind] = 1

        self.targets = targets

    @classmethod
    def getRandomAngle(cls, forbidden_angles):
        MIN_ANGLE = -90
        MAX_ANGLE = 90
        STEP = 5

        angles_list = list(range(MIN_ANGLE, MAX_ANGLE + STEP, STEP))

        #Remove forbidden angles
        possible_angles = [angle for angle in angles_list if angle not in forbidden_angles]

        return random.choice(possible_angles)

    @classmethod
    def getAngleIdx(cls, angle):
        return int((angle+90)/5)

    @classmethod
    def getAngleFromIdx(cls, angle_idx):
        return angle_idx*5-90

    @classmethod
    def getBinauralSound(cls, audio_in, brir, pos_def, angle):
        assert isinstance(audio_in, numpy.ndarray)
        assert isinstance(brir, numpy.ndarray)
        assert isinstance(pos_def, numpy.ndarray)

        if (angle > 90 or angle < -90):
            raise Exception('Only angles between -90 and 90 are supported')
        if (angle < 0):
            angle = angle + 360

        # Find closest angle in pos_def
        angles = pos_def[:, 0]
        angle_idx = DataEntry.find_closest_idx_in_array(angles, angle)

        brir_direction = brir[angle_idx, :, :]
        channel1 = scipy.signal.convolve(audio_in, brir_direction[0, :], 'same')
        channel2 = scipy.signal.convolve(audio_in, brir_direction[1, :], 'same')
        audio_out = numpy.vstack((channel1, channel2)).T

        return audio_out

    @classmethod
    def find_closest_idx_in_array(cls, array, val):
        assert isinstance(array, numpy.ndarray)

        min_idx = numpy.abs(array - val).argmin()
        return min_idx

    def updateFeaturesFromResSignal(self):
        self.features = numpy.array([])

        mfcc = features.getMFCC(self.res_signal)
        self.features = mfcc

        ild = features.getILD(self.res_signal)
        self.features = numpy.hstack((self.features, ild))

        ipd = features.getIPD(self.res_signal)
        self.features = numpy.hstack((self.features, ipd))

    def saveDataSetRecord(self):
        folder = self.save_folder
        if(os.path.isdir(folder) == False):
            os.makedirs(folder)

        #Save original wav files
        for ind in range(len(self.signals)):
            save_path = os.path.join(folder, 'Original_{0}.wav'.format(ind+1))
            scipy.io.wavfile.write(save_path, int(parameters.SAMPLE_RATE_HZ), self.signals[ind])

        #Save mixture wav file
        save_path = os.path.join(folder, 'Mixture.wav')
        scipy.io.wavfile.write(save_path, int(parameters.SAMPLE_RATE_HZ), self.res_signal)

        #Save Cochleagram images
        for ind in range(len(self.signals)):
            cgram = features.getCochleagram(self.signals[ind])
            fig = plt.figure()
            plt.imshow(cgram.T, extent=(0, parameters.SIGNAL_LENGTH_SEC*1000, cgram.shape[1], 0), aspect='auto')
            plt.title('Cochleagram plot for original signal {0}'.format(ind+1))
            plt.xlabel('Time[ms]')
            plt.ylabel('Filterbank index')
            save_path = os.path.join(folder, 'Origin_Cochleagram_{0}'.format(ind+1))
            plt.savefig(save_path)
            plt.close(fig)

        #Cgram for mixture
        cgram = features.getCochleagram(self.res_signal)
        fig = plt.figure()
        plt.imshow(cgram.T, extent=(0, parameters.SIGNAL_LENGTH_SEC*1000, cgram.shape[1], 0), aspect='auto')
        plt.title('Cochleagram plot for mixture signal')
        plt.xlabel('Time[ms]')
        plt.ylabel('Filterbank index')
        save_path = os.path.join(folder, 'Mixture_Cochleagram')
        plt.savefig(save_path)
        plt.close(fig)

        #Save mixed ibm
        mixed_ibm = self.dnnTargetToMixedIbm(self.targets)
        fig = plt.figure()
        plt.imshow(parameters.NUM_OF_DIRECTIONS - mixed_ibm.T,
                   extent=(0, parameters.SIGNAL_LENGTH_SEC * 1000, cgram.shape[1], 0), aspect='auto')
        plt.title('Mixed ibm plot for mixture signal')
        plt.xlabel('Time[ms]')
        plt.ylabel('Filterbank index')
        save_path = os.path.join(folder, 'Mixed_ibm')
        plt.savefig(save_path)
        plt.close(fig)

        # Get ibms for each original signal and save it
        (unique_ibms, angles) = self.mixedIbmToIbms(mixed_ibm)
        for ind in range(len(unique_ibms)):
            ibm = unique_ibms[ind]
            fig = plt.figure()
            plt.imshow(ibm.T, extent=(0, parameters.SIGNAL_LENGTH_SEC * 1000, parameters.CGRAM_NUM_CHANNELS, 0),
                       aspect='auto')
            plt.title('IBM plot for signal {0}'.format(ind + 1))
            plt.xlabel('Time[ms]')
            plt.ylabel('Filterbank index')
            save_path = os.path.join(self.save_folder, 'Original_{0}_IBM'.format(ind + 1))
            plt.savefig(save_path)
            plt.close(fig)

    @classmethod
    def dnnTargetToMixedIbm(cls, dnn_target):
        mixed_ibm = numpy.zeros((dnn_target[0].shape[0], len(dnn_target)))
        for time_dim in range(mixed_ibm.shape[0]):
            for freq_dim in range(mixed_ibm.shape[1]):
                mixed_ibm[time_dim, freq_dim] = dnn_target[freq_dim][time_dim, :].argmax()

        return mixed_ibm

    @classmethod
    def mixedIbmToIbms(cls, mixed_ibm):
        assert isinstance(mixed_ibm, numpy.ndarray)

        (angle_ind, angle_count) = numpy.unique(mixed_ibm, return_counts=True)
        ibms = []
        angles = []
        for ind in range(len(angle_ind)):
            if(angle_count[ind] < parameters.MIXED_IBM_IDENTIFICATION_TH):
                continue

            #Ignore the ibm of the noise
            if(angle_ind[ind] >= parameters.NUM_OF_DIRECTIONS):
                continue

            ibm = numpy.zeros(mixed_ibm.shape)
            ibm[mixed_ibm==angle_ind[ind]] = 1
            ibms.append(ibm)
            angles.append(cls.getAngleFromIdx(angle_ind[ind]))

        return (ibms, angles)

    def estimateNetPerformance(self, net, save=True):
        assert isinstance(net, Model)

        net_output = net.predict(self.features)

        #Get predicted mixed ibm and save it
        mixed_ibm = self.dnnTargetToMixedIbm(net_output)
        if(save == True):
            fig = plt.figure()
            plt.imshow(parameters.NUM_OF_DIRECTIONS - mixed_ibm.T,
                       extent=(0, parameters.SIGNAL_LENGTH_SEC * 1000, parameters.CGRAM_NUM_CHANNELS, 0), aspect='auto')
            plt.title('Predicted mixed ibm plot for mixture signal')
            plt.xlabel('Time[ms]')
            plt.ylabel('Filterbank index')
            save_path = os.path.join(self.save_folder, 'Predicted_Mixed_ibm')
            plt.savefig(save_path)
            plt.close(fig)

        #Reconstruct signals
        (ibms, angles) = self.mixedIbmToIbms(mixed_ibm)
        if(save == True):
            for ind in range(len(ibms)):
                signal = features.applyIbmToSignal(self.res_signal[:,0], ibms[ind])
                save_path = os.path.join(self.save_folder, 'estimated_signal_{0}.wav'.format(ind))
                scipy.io.wavfile.write(save_path, int(parameters.SAMPLE_RATE_HZ), signal)

        #Calculate performance
        performance = {}

        source_md = len([a for a in self.angles if a not in angles])/len(self.angles)
        performance['source_md'] = source_md

        source_fa = len([a for a in angles if a not in self.angles])/len(angles)
        performance['source_fa'] = source_fa

        return performance