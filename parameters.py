# This file contains all parameters of the project

#DNN parameters
DNN_HIDDEN_LAYER_SIZE = [1024, 1024, 1024]
MAX_EPOCHS_TRAIN = 25

#Audio parameters
SAMPLE_RATE_HZ = float(16e3)
WINDOW_LENGTH_MS = float(20)
WINDOW_STEP_MS = float(10)
assert ((WINDOW_LENGTH_MS/1000.0)*SAMPLE_RATE_HZ).is_integer() == True
assert ((WINDOW_STEP_MS/1000.0)*SAMPLE_RATE_HZ).is_integer() == True

WINDOW_SIZE_SAMPLES = int((WINDOW_LENGTH_MS / 1000.0) * SAMPLE_RATE_HZ)
WINDOW_STEP_SAMPLES = int((WINDOW_STEP_MS / 1000.0) * SAMPLE_RATE_HZ)

SIGNAL_LENGTH_SEC = 2.4
NUM_OF_SOURCES_IN_SIGNAL = 2
NUM_OF_TRAIN_SIGNALS = 200
NUM_OF_TEST_SIGNALS = 50

NUM_OF_DIRECTIONS = 36

INPUT_SIGNAL_RMS = 1e-2

#Cochleagram parameters
CGRAM_MIN_FREQ = 80
CGRAM_MAX_FREQ = 5000
CGRAM_NUM_CHANNELS = 32
CGRAM_NOISE_TH_dB = -10

#Spectrogram parameters
SGRAM_NOISE_TH_dB = -5
SGRAM_NUM_CHANNELS = int(WINDOW_SIZE_SAMPLES/2+1)

#MFCC parameters
MFCC_NUM_COEFF = 31
MFCC_MIN_FREQ = CGRAM_MIN_FREQ
MFCC_MAX_FREQ = CGRAM_MAX_FREQ

#Folders
OUTPUT_FOLDER = r'..\Results'
BRIR_FILE = r'..\Database\BRIR\UniS_Anechoic_BRIR_16k.sofa'
TRAIN_SENTENCES_FOLDER = r'..\Database\Clean\Train_Data'
TEST_SENTENCES_FOLDER = r'..\Database\Clean\Test_Data'

#IBM Parameters
MIXED_IBM_IDENTIFICATION_TH = 1000
