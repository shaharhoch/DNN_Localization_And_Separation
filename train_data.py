import parameters
import numpy
import data_entry

class TrainData():

    def __init__(self):
        fs = parameters.SAMPLE_RATE_HZ
        sig_time = parameters.SIGNAL_LENGTH_SEC
        win_len = parameters.WINDOW_SIZE_SAMPLES
        win_step = parameters.WINDOW_STEP_SAMPLES
        num_of_windows_in_single = int(((sig_time * fs) - win_len) / win_step + 1)
        num_of_windows = int(num_of_windows_in_single * parameters.NUM_OF_TRAIN_SIGNALS)

        self.data_ind = 0

        # Input input feature array
        self.train_input = numpy.zeros((num_of_windows, parameters.getSizeOfFeatureVec()))

        # Init output target list of vectors
        self.train_targets = []
        for ind in range(parameters.SGRAM_NUM_CHANNELS):
            self.train_targets.append(numpy.zeros((num_of_windows, parameters.NUM_OF_DIRECTIONS + 1)))

        #Init mean and variance to values that won't do anything
        self.mean = numpy.zeros(parameters.getSizeOfFeatureVec())
        self.std = numpy.ones(parameters.getSizeOfFeatureVec())

        #This is used to lock data after normalization, because we can't add data after we normalize it
        self.data_locked = False


    def addDataEntry(self, entry):
        assert isinstance(entry, data_entry.DataEntry)
        assert (self.data_ind <= self.train_input.shape[0]-entry.features.shape[0])
        assert (self.data_locked == False)

        # Update train inputs
        self.train_input[self.data_ind:self.data_ind+entry.features.shape[0], :] = entry.features

        # Update train targets list
        for ind in range(len(self.train_targets)):
            self.train_targets[ind][self.data_ind:self.data_ind+entry.targets[ind].shape[0], :] = entry.targets[ind]

        self.data_ind += entry.features.shape[0]

    def isDataFull(self):
        assert  (self.data_ind <= self.train_input.shape[0])
        return (self.data_ind == self.train_input.shape[0])

    def isDataLocked(self):
        return self.data_locked

    def getTrainInputs(self):
        return self.train_input[0:self.data_ind]

    def getTrainTargets(self):
        if(self.isDataFull()):
            return self.train_targets

        targets = []
        for ind in range(len(self.train_targets)):
            targets.append(self.train_targets[ind][0:self.data_ind])

        return targets

    def meanVarianceNormalize(self):
        self.mean = numpy.mean(self.train_input, axis=0)
        self.std = numpy.std(self.train_input, axis=0)
        # If there are features with std==0 this means they are constant, which means that they will be 0.
        # In this case we choose std=1, because if in the training phase the features are different we don't want to scale them,
        # becasue we don't really know their distribution
        self.std[self.std == 0] = 1
        self.train_input = (self.train_input - self.mean) / self.std

        #Lock data after normalization
        self.data_locked = True
