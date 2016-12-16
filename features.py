import python_speech_features
import numpy
import parameters
import numpy.fft
import sys
import scipy.signal
import scipy.io

def getMFCC(audio_in):
    assert isinstance(audio_in, numpy.ndarray)
    assert (audio_in.size % parameters.WINDOW_STEP_SAMPLES) == 0

    # Get single channel if input is dual channel
    if(len(audio_in.shape) > 1):
        audio_in = audio_in[:, 0]

    mfcc_features = python_speech_features.mfcc(audio_in, samplerate=parameters.SAMPLE_RATE_HZ, \
                                       winlen=parameters.WINDOW_LENGTH_MS*(1e-3), winstep=parameters.WINDOW_STEP_MS*(1e-3),\
                                       numcep=parameters.MFCC_NUM_COEFF+1, lowfreq=parameters.MFCC_MIN_FREQ, \
                                       highfreq=parameters.MFCC_MAX_FREQ, nfilt=parameters.MFCC_NUM_COEFF*2)

    assert isinstance(mfcc_features, numpy.ndarray)
    return mfcc_features[:,1:mfcc_features.shape[1]]

def divideSignalToWindows(signal):
    fs = parameters.SAMPLE_RATE_HZ
    win_size = int((parameters.WINDOW_LENGTH_MS/1000.0)*parameters.SAMPLE_RATE_HZ)
    win_step = int((parameters.WINDOW_STEP_MS/1000.0)*parameters.SAMPLE_RATE_HZ)

    num_of_windows = int((float(len(signal)-parameters.WINDOW_SIZE_SAMPLES)/float(parameters.WINDOW_STEP_SAMPLES))+1)
    divided_signal = numpy.zeros([num_of_windows, win_size])
    for i in range(num_of_windows):
        start_ind = i*win_step
        #Stop ind is actually the stop index +1, because python indexing goes up to end index -1
        stop_ind = start_ind+win_size
        divided_signal[i, :] = signal[start_ind:stop_ind]

    return divided_signal

def erb2hz(erb):
    hz = (10**(erb / 21.4) - 1) / 4.37e-3
    return hz

def hz2erb(hz):
    erb = 21.4 * numpy.log10(4.37e-3 * hz + 1)
    return erb

'''
Compute loudness level in Phons on the basis of equal-loudness functions.
It accounts a middle ear effect and is used for frequency-dependent gain adjustments.
This function uses linear interpolation of a lookup table to compute the loudness level,
in phons, of a pure tone of frequency freq using the reference curve for sound
pressure level dB. The equation is taken from section 4 of BS3383.
'''
def loudness(freq):
    '''
    Stores parameters of equal-loudness functions from BS3383,"Normal equal-loudness level
    contours for pure tones under free-field listening conditions", table 1.
    f (or ff) is the tone frequency, af and bf are frequency-dependent coefficients, and
    tf is the threshold sound pressure level of the tone, in dBs
    '''
    data_file = scipy.io.loadmat('f_af_bf_cf.mat')
    ff = data_file['ff']
    ff = numpy.reshape(ff, ff.size)
    af = data_file['af']
    af = numpy.reshape(af, af.size)
    bf = data_file['bf']
    bf = numpy.reshape(bf, bf.size)
    cf = data_file['cf']
    cf = numpy.reshape(cf, cf.size)

    dB = 60
    if (freq < 20 or freq > 12500):
        raise Exception('Cannot compute loudness for given frequency')

    i = 0
    while (ff[i] < freq):
        i = i + 1

    afy = af[i - 1] + (freq - ff[i - 1]) * (af[i] - af[i - 1]) / (ff[i] - ff[i - 1])
    bfy = bf[i - 1] + (freq - ff[i - 1]) * (bf[i] - bf[i - 1]) / (ff[i] - ff[i - 1])
    cfy = cf[i - 1] + (freq - ff[i - 1]) * (cf[i] - cf[i - 1]) / (ff[i] - ff[i - 1])
    loud = 4.2 + afy * (dB - cfy) / (1 + bfy * (dB - cfy))
    return loud

def getGammatoneCenterFreq():
    # Find center frequencies- they are equally spaced on the erb scale
    min_erb = hz2erb(parameters.CGRAM_MIN_FREQ)
    max_erb = hz2erb(parameters.CGRAM_MAX_FREQ)
    center_freq_erb = numpy.linspace(min_erb, max_erb, parameters.CGRAM_NUM_CHANNELS)
    # Get center frequencies in hz. They are named fc in the gammatone definition
    fc = erb2hz(center_freq_erb)

    return fc

def getGammatoneFir():
    GT_FILTER_ORDER = 4
    GT_LENGTH_MS = 128  # Not sure what to use here. Saw this value is being used.
    GT_LENGTH_SAMPLES = int(GT_LENGTH_MS * 1e-3 * parameters.SAMPLE_RATE_HZ)

    fc = getGammatoneCenterFreq()
    # Get each filters bandwidth. named b in the gammatone definition
    b = 1.019 * 24.7 * (4.37 * fc / 1000 + 1)

    # Build filters impulse responses and filter the signal with them
    time = numpy.array(range(GT_LENGTH_SAMPLES)) / parameters.SAMPLE_RATE_HZ
    gt_fir = numpy.zeros((parameters.CGRAM_NUM_CHANNELS, GT_LENGTH_SAMPLES))
    for channel in range(parameters.CGRAM_NUM_CHANNELS):
        gain = 10 ** ((loudness(fc[channel]) - 60) / 20) / 3 * (2 * numpy.pi * b[channel] / parameters.SAMPLE_RATE_HZ) ** 4

        # In the impulse response we multiply the time by the sampling frequency to make it unit-less
        gt_fir[channel,:] = gain*((time*parameters.SAMPLE_RATE_HZ)**(GT_FILTER_ORDER-1))*numpy.exp(-2*numpy.pi*b[channel]*time)*numpy.cos(2*numpy.pi*fc[channel]*time)

    return gt_fir

def applyGammatoneFilterbank(signal_in):
    gt_fir = getGammatoneFir()
    gt_filtered = numpy.zeros((parameters.CGRAM_NUM_CHANNELS, len(signal_in)))
    for channel in range(parameters.CGRAM_NUM_CHANNELS):
        gt_filtered[channel,:] = scipy.signal.convolve(signal_in, gt_fir[channel,:], 'same')

    return gt_filtered

def applyIbmToSignal(signal_in, ibm=None):
    assert isinstance(signal_in, numpy.ndarray)
    if(ibm == None):
        num_of_windows = int((float(len(signal_in) - parameters.WINDOW_SIZE_SAMPLES) / float(parameters.WINDOW_STEP_SAMPLES)) + 1)
        ibm = numpy.ones((num_of_windows, parameters.CGRAM_NUM_CHANNELS))
    assert isinstance(ibm, numpy.ndarray)

    gt_filtered = applyGammatoneFilterbank(signal_in)
    gt_fir = getGammatoneFir()
    fc = getGammatoneCenterFreq()

    #Get raised cosine window
    # Get time vector in range [0,1] in size of window
    time = numpy.array(range(parameters.WINDOW_SIZE_SAMPLES))/parameters.WINDOW_SIZE_SAMPLES
    cos_win = (1+numpy.cos(2*numpy.pi*time-numpy.pi))/2

    signal_out = numpy.zeros(signal_in.shape)
    for channel in range(parameters.CGRAM_NUM_CHANNELS):
        mid_ear_coeff = 10**((loudness(fc[channel])-60)/20)
        channel_signal = gt_filtered[channel,::-1]/mid_ear_coeff #Flip signal and divide by mid_ear_coeffs
        channel_signal = scipy.signal.convolve(channel_signal, gt_fir[channel,:], 'same')
        channel_signal = channel_signal[::-1]/mid_ear_coeff #Flip signal and divide by mid_ear_coeffs again

        #Get ibm weight
        win_len = parameters.WINDOW_SIZE_SAMPLES
        win_shift = parameters.WINDOW_STEP_SAMPLES
        weight = numpy.zeros(len(signal_in))

        weight[0:win_len/2] = ibm[0,channel]*cos_win[win_len/2:]
        for frame_ind in range(1,ibm.shape[0]):
            mid_ind = win_shift*frame_ind
            weight[mid_ind-win_len/2:mid_ind+win_len/2] = weight[mid_ind-win_len/2:mid_ind+win_len/2] + ibm[frame_ind,channel]*cos_win
        signal_out = signal_out + weight*channel_signal

    return signal_out

def getCochleagram(audio_in):
    assert isinstance(audio_in, numpy.ndarray)
    assert (audio_in.size % parameters.WINDOW_STEP_SAMPLES) == 0

    #Get single channel if input is dual channel
    if(len(audio_in.shape) > 1):
        audio_in = audio_in[:,0]

    gt_filtered = applyGammatoneFilterbank(audio_in)
    assert isinstance(gt_filtered, numpy.ndarray)

    # Get Cochleagram from gammatone filtered signals
    num_of_windows = int((float(len(audio_in) - parameters.WINDOW_SIZE_SAMPLES) / float(parameters.WINDOW_STEP_SAMPLES)) + 1)

    cgram = numpy.zeros([num_of_windows,parameters.CGRAM_NUM_CHANNELS])
    for i in range(parameters.CGRAM_NUM_CHANNELS):
        divided_channel = divideSignalToWindows(gt_filtered[i,:])
        channel_energy = numpy.linalg.norm(divided_channel, axis=1)
        channel_energy = channel_energy*channel_energy #Cochleagram needs the energy, which is the norm squared
        cgram[:,i] = channel_energy

    return cgram

def getIPD(audio_in):
    assert isinstance(audio_in, numpy.ndarray)
    assert (audio_in.shape[1] == 2)

    channel1 = divideSignalToWindows(audio_in[:,0])
    channel2 = divideSignalToWindows(audio_in[:,1])

    channel1_fft = numpy.fft.fft(channel1, axis=1)
    channel2_fft = numpy.fft.fft(channel2, axis=1)

    #Avoid dividing by zero- change every elemnt equal to zero to a very small value
    channel2_fft[channel2_fft == 0] = sys.float_info.min
    return numpy.angle(channel1_fft/channel2_fft)

def getILD(audio_in):
    assert isinstance(audio_in, numpy.ndarray)
    assert (audio_in.shape[1] == 2)

    channel1 = divideSignalToWindows(audio_in[:, 0])
    channel2 = divideSignalToWindows(audio_in[:, 1])

    channel1_fft = numpy.fft.fft(channel1, axis=1)
    channel2_fft = numpy.fft.fft(channel2, axis=1)

    # Avoid dividing by zero- change every elemnt equal to zero to a very small value
    channel2_fft[channel2_fft == 0] = sys.float_info.min

    amplitude_ratio = numpy.abs(channel1_fft/channel2_fft)

    #Avoid taking log of zero
    amplitude_ratio[amplitude_ratio == 0] = sys.float_info.min
    return 20*numpy.log10(amplitude_ratio)