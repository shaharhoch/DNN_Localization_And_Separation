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

    mfcc_features = python_speech_features.mfcc(audio_in, samplerate=parameters.SAMPLE_RATE_HZ,
                                       winlen=parameters.WINDOW_LENGTH_MS*(1e-3), winstep=parameters.WINDOW_STEP_MS*(1e-3),
                                       numcep=parameters.MFCC_NUM_COEFF+1, lowfreq=parameters.MFCC_MIN_FREQ,
                                       highfreq=parameters.MFCC_MAX_FREQ, nfilt=parameters.MFCC_NUM_COEFF*2)

    assert isinstance(mfcc_features, numpy.ndarray)
    return mfcc_features[:,1:mfcc_features.shape[1]]

def divideSignalToWindows(signal):
    fs = parameters.SAMPLE_RATE_HZ
    win_size = parameters.WINDOW_SIZE_SAMPLES
    win_step = parameters.WINDOW_STEP_SAMPLES

    num_of_windows = int((float(len(signal)-win_size)/float(win_step))+1)
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
    if(ibm is None):
        num_of_windows = int((float(len(signal_in) - parameters.WINDOW_SIZE_SAMPLES) / float(parameters.WINDOW_STEP_SAMPLES)) + 1)
        ibm = numpy.ones((num_of_windows, parameters.SGRAM_NUM_CHANNELS))
    assert isinstance(ibm, numpy.ndarray)

    stft = getStft(signal_in)
    masked_stft = stft*ibm
    return getIstft(masked_stft)

def getCochleagram(audio_in):
    assert isinstance(audio_in, numpy.ndarray)
    assert (audio_in.size % parameters.WINDOW_STEP_SAMPLES) == 0

    # Get single channel if input is dual channel
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

    channel1_fft = getStft(audio_in[:,0])
    channel2_fft = getStft(audio_in[:,1])

    #Avoid dividing by zero- change every elemnt equal to zero to a very small value
    channel2_fft[channel2_fft == 0] = sys.float_info.min
    return numpy.angle(channel1_fft/channel2_fft)

def getILD(audio_in):
    assert isinstance(audio_in, numpy.ndarray)
    assert (audio_in.shape[1] == 2)

    channel1_fft = getStft(audio_in[:, 0])
    channel2_fft = getStft(audio_in[:, 1])

    # Avoid dividing by zero- change every elemnt equal to zero to a very small value
    channel2_fft[channel2_fft == 0] = sys.float_info.min

    amplitude_ratio = numpy.abs(channel1_fft/channel2_fft)

    #Avoid taking log of zero
    amplitude_ratio[amplitude_ratio == 0] = sys.float_info.min
    return 20*numpy.log10(amplitude_ratio)

def getMV(audio_in):
    MV_FEATURES_PER_BIN = 4

    assert  isinstance(audio_in, numpy.ndarray)

    channel1_fft = getStft(audio_in[:, 0])
    channel2_fft = getStft(audio_in[:, 1])

    mv = numpy.zeros([channel1_fft.shape[0], MV_FEATURES_PER_BIN*channel1_fft.shape[1]])
    for channel_ind in range(channel1_fft.shape[1]):
        x_vec = numpy.column_stack((channel1_fft[:, channel_ind], channel2_fft[:, channel_ind]))
        x_norm = numpy.array(2*[numpy.linalg.norm(x_vec, axis=1)]).T
        # This comes to avoid dividing by zero. The normalization will give zero in any case, because x_vec is zero
        x_norm[x_norm == 0] = sys.float_info.min
        x_gal = x_vec/x_norm

        # Get W matrix
        W_matrix_mean = numpy.zeros(2*[x_gal.shape[1]], dtype=numpy.complex128)
        for time_ind in range(x_gal.shape[0]):
            W_matrix_mean += numpy.outer(x_gal[time_ind,:], x_gal[time_ind,:].conj().T)
        W_matrix_mean /= x_gal.shape[0]

        # Take only the eigen-vectors, ignore the eigen-values
        # We use transpose because the eigen-vectors should be the rows of W, but the eig function returns them as columns
        W = numpy.linalg.eig(W_matrix_mean)[1].T

        # Get MV features
        for time_ind in range(x_gal.shape[0]):
            cur_mv = W.dot(x_gal[time_ind,:])

            mv_norm = numpy.linalg.norm(cur_mv)
            # This comes to avoid dividing by zero
            if(mv_norm != 0):
                cur_mv /= mv_norm

            start_ind = MV_FEATURES_PER_BIN*channel_ind
            stop_ind = start_ind + MV_FEATURES_PER_BIN
            mv[time_ind, start_ind:stop_ind] = numpy.array([cur_mv[0].real, cur_mv[0].imag, cur_mv[1].real, cur_mv[1].imag])

    return mv

def getStft(in_signal):
    assert isinstance(in_signal, numpy.ndarray)
    assert len(in_signal.shape) == 1

    win_size = parameters.WINDOW_SIZE_SAMPLES
    win_shift = parameters.WINDOW_STEP_SAMPLES
    window = numpy.hanning(win_size)
    stft = numpy.array([numpy.fft.rfft(window * in_signal[i:i+win_size])
                     for i in range(0, len(in_signal)-win_size+1, win_shift)])
    return stft

def getSpectrogram(in_signal):
    return numpy.abs(getStft(in_signal))

def getIstft(in_stft):
    assert  isinstance(in_stft, numpy.ndarray)

    win_size = parameters.WINDOW_SIZE_SAMPLES
    win_shift = parameters.WINDOW_STEP_SAMPLES

    out = numpy.zeros(win_shift*in_stft.shape[0] + (win_size-win_shift))

    for n, i in enumerate(range(0, len(out) - win_size + 1, win_shift)):
        out[i:i + win_size] += numpy.fft.irfft(in_stft[n,:])

    return out