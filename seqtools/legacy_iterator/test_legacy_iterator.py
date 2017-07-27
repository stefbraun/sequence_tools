from __future__ import print_function
from legacy_iterator import LegacyBatchIterator, check_zmuv
import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import zmq

# Set parameters
max_frame_size = 20000
shuffle_type = 'none'
normalizations = ['none', 'epoch', 'batch', 'sample']
shuffle_types = ['none', 'shuffle', 'high_throughput', 'exp']

pem = 0  # 1
dataset = 'test_dev93'  # test_dev93
epoch = 1  # 1
SNR = [111]  # [111]
gpu = 'test'  # 'test'
mode = 'dbg'  # 'dbg'
debug = 0  # 0

# Load key verificators: reference
metadata_dir = '/media/stefbraun/ext4/Dropbox/dataset/WSJ/wsj_reference/'
transcripts_int = pkl.load(open(metadata_dir + 'ref_dev_93_transcripts_int.pkl', 'rb'))
transcripts_lens = pkl.load(open(metadata_dir + 'ref_dev_93_transcripts_lens.pkl', 'rb'))

if pem == 1:
    # Define the socket using the "Context"
    context = zmq.Context()
    socket = get_socket_from_gpu('test')
    sock = context.socket(zmq.REQ)
    sock.connect("tcp://127.0.0.1:{}".format(socket))

    # Send a "message" using the socket
    work = [(dataset, epoch, SNR, gpu, mode, debug)]
    sock.send_pyobj(work)
    path = sock.recv_pyobj()
    path = path['dbg']

else:
    # path = '/home/stefbraun/data/wsj_reference/ref_train_si84_clean.h5'
    path = '/media/stefbraun/ext4/Dropbox/dataset/WSJ/wsj_reference/ref_dev_93_clean.h5'
    print('Precalculated h5 is used:  {}'.format(path))

# Initialize batch iterator
d = LegacyBatchIterator()
epoch = 1

err = 0

# Check frame cache
frames_ep_bw = dict()
frames_ep = dict()
padded_frames_ep = dict()
checksums = {'none': 612536, 'shuffle': 655983, 'high_throughput': 406819, 'exp': 497318}
for s_type in shuffle_types:
    print(s_type)
    batch_lens = []
    padded_frames = []
    batch_lens_bw = []
    for bX, b_lenX, maskX, bY, b_lenY, batch_monitor in d.flow(epoch=epoch, h5=path,
                                                               shuffle_type=s_type,
                                                               max_frame_size=max_frame_size,
                                                               normalization='sample', enable_gauss=0):
        padded_frames.append(bX.shape[0] * bX.shape[1])
        batch_lens.extend(b_lenX)
        batch_lens_bw.append(np.sum(b_lenX))

    if np.sum(padded_frames) != checksums[s_type]:
        print('Frame cache error')
        err += 1
    padded_frames_ep[s_type] = padded_frames
    frames_ep[s_type] = batch_lens
    frames_ep_bw[s_type] = batch_lens_bw
    if np.sum(frames_ep[s_type]) != batch_monitor['frames']:
        print('Batch monitor frame count is wrong')
        err += 1

    if np.sum(padded_frames_ep[s_type]) != batch_monitor['padded_frames']:
        print('Batch monitor padded frame count is wrong')
        err += 1

# Check gaussian noise
print('Gaussian noise')
ep_dict = dict()
for noise in [0, 1]:
    if noise ==0:
        eg =0
    if noise ==1:
        eg=0.6
    ep_features = []
    for bX, b_lenX, maskX, bY, b_lenY, batch_monitor in d.flow(epoch=epoch, h5=path,
                                                               shuffle_type='shuffle',
                                                               max_frame_size=max_frame_size,
                                                               normalization='sample', enable_gauss=eg):
        for i, sample in enumerate(bX):
            ep_features.append(sample[:b_lenX[i], :])
        if batch_monitor['batch_no'] > 5:
            break
    ep_dict['{}'.format(noise)] = ep_features

for sample, noisy_sample in zip(ep_dict['0'], ep_dict['1']):
    noise = noisy_sample - sample
    if not np.isclose(0, np.mean(noise), atol=1e-2):
        print('Noise mean is {} instead of 0'.format(np.mean(noise)))
        err = err + 1
    if not np.isclose(0.6, np.std(noise), atol=1e-2):
        print('Noise std is {} instead of 0.6'.format(np.std(noise)))
        err = err + 1

plt.figure()
plt.subplot(311)
noise_mat = ep_dict['1'][0] - ep_dict['0'][0]
plt.imshow(noise_mat.T, interpolation='none', cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Gaussian noise, mean: {} , std: {}'.format(np.mean(noise_mat), np.std(noise_mat)))
plt.subplot(312)
sample = ep_dict['0'][0]
plt.imshow(sample.T, interpolation='none', cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Sample without noise, mean: {} , std: {}'.format(np.mean(sample), np.std(sample)))
plt.subplot(313)
sample = ep_dict['1'][0]
plt.imshow(sample.T, interpolation='none', cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Sample with noise, mean: {} , std: {}'.format(np.mean(sample), np.std(sample)))

plt.figure()
count, bins, ignored = plt.hist(noise_mat, 30, normed=True)
plt.plot(bins, 1 / (0.6 * np.sqrt(2 * np.pi)) * np.exp(- (bins - 0) ** 2 / (2 * 0.6 ** 2)), linewidth=2, color='r')
plt.grid()
plt.title('Histogram of noise')

# Check rest
norm_samples = dict()
norm_ep = dict()
plt.figure()
for norm in normalizations:
    times = dict()
    ep_features = []
    batch_features = []
    sample_features = []
    batch_lens = []

    labels = []
    label_lens = []

    all_keys = []

    padded_frames = []
    print(norm)
    cnt = 0
    for bX, b_lenX, maskX, bY, b_lenY, batch_monitor in d.flow(epoch=epoch, h5=path,
                                                               shuffle_type=shuffle_type,
                                                               max_frame_size=max_frame_size,
                                                               normalization=norm, enable_gauss=0):

        curr_batch_features = []
        for i, sample in enumerate(bX):
            ep_features.extend(sample[:b_lenX[i], :])
            curr_batch_features.extend(sample[:b_lenX[i], :])
            sample_features.append(sample[:b_lenX[i], :])
            if (cnt == 5 and i == 0):
                this_key = batch_monitor['epoch_keys'][-1][0]

                plt_sample = sample[:b_lenX[i], :]
                if norm == 'none':
                    plt.subplot(411)
                    plt.imshow(plt_sample.T, interpolation='none', cmap='viridis', aspect='auto')
                    plt.title('No Normalization {}'.format(this_key))
                    plt.colorbar()
                    norm_samples[norm] = plt_sample

                if norm == 'epoch':
                    plt.subplot(412)
                    plt.imshow(plt_sample.T, interpolation='none', cmap='viridis', aspect='auto')
                    plt.title('Epoch Normalization {}'.format(this_key))
                    plt.colorbar()
                    norm_samples[norm] = plt_sample

                if norm == 'batch':
                    plt.subplot(413)
                    plt.imshow(plt_sample.T, interpolation='none', cmap='viridis', aspect='auto')
                    plt.title('Batch Normalization {}'.format(this_key))
                    plt.colorbar()
                    norm_samples[norm] = plt_sample

                if norm == 'sample':
                    plt.subplot(414)
                    plt.imshow(plt_sample.T, interpolation='none', cmap='viridis', aspect='auto')
                    plt.title('Sample Normalization {}'.format(this_key))
                    plt.colorbar()
                    norm_samples[norm] = plt_sample

        batch_features.append(curr_batch_features)

        batch_lens.extend(b_lenX)

        labels.extend(bY)
        label_lens.extend(b_lenY)

        # Check batch normalization
        if norm == 'batch':
            check_feats = np.asarray(curr_batch_features)
            err = err + check_zmuv(check_feats, 'batch')

        # Check sample normalization
        if norm == 'sample':
            for i, sample in enumerate(bX):
                non_padded = sample[:b_lenX[i], :]
                check_feats = np.asarray(non_padded)
                err = err + check_zmuv(check_feats, 'sample')

        # Check max frame size
        if bX.shape[0] * bX.shape[1] > max_frame_size:
            print('Max_frame size error')
            err = err + 1

        # Check keys
        ver_labels = []
        ver_label_lens = []
        for ky in batch_monitor['epoch_keys'][-1]:
            all_keys.append(ky)
            ver_labels.extend(transcripts_int[ky])
            ver_label_lens.extend(transcripts_lens[ky])

            if ky in transcripts_int.keys() == 0:
                print('Key does not belong to test_dev93')
                err = err + 1

        if np.sum(bY) != np.sum([i + 1 for i in ver_labels]):
            print('Labels checksum incorrect')
            err = err + 1

        if np.sum(b_lenY) != np.sum(ver_label_lens):
            print('Label lens checksum incorrect')
            err = err + 1

        cnt = cnt + 1

    feats = np.asarray(ep_features)
    norm_ep[norm] = feats

    if norm == 'None':
        if np.isclose(np.sum(feats), 165180752) == 0:
            print('Features checksum incorrect')
            err += 1

    if norm == 'epoch':
        err = err + check_zmuv(feats, 'epoch')

    all_keys = []
    for minibatch in batch_monitor['epoch_keys']:
        all_keys.extend(minibatch)
    if len(np.unique(all_keys)) != 503:
        print('Not enough unique keys')
        err += 1

    if np.sum(batch_lens) != feats.shape[0]:
        print('Batch lengths incorrect')
        err += 1

    if np.sum(label_lens) != 48720:  # equivalent to fuel
        print('Label_lens checksum incorrect')
        err += 1

    if np.sum(labels) != 1678406:  # equivalent to fuel
        print('Labels checksum incorrect')
        err += 1

    if feats.shape != (390177, 123):
        print('Features array shape incorrect')
        err += 1

    if len(np.unique(all_keys)) != 503:
        print('Not all keys were processed.')
        err += 1

    if np.mean(batch_monitor['snr']) != 83.25:
        print('SNR values are not clean')
        err += 1

print('Frame cache')
var_keys = ['method', 'part', 'frames', 'padded_frames', 'mean(frames)', 'std(frames)', 'min(frames)', 'max(frames)',
            'max(padded_frames)']
print('{:16} {} {} {} {} {} {} {} {}'.format(*var_keys))
for key in shuffle_types:
    varis = [key, float(np.sum(frames_ep[key])) / np.sum(padded_frames_ep[key]), np.sum(frames_ep[key]),
             np.sum(padded_frames_ep[key]), np.mean(frames_ep_bw[key]), np.std(frames_ep_bw[key]),
             np.min(frames_ep_bw[key]), np.max(frames_ep_bw[key]), np.max(padded_frames_ep[key])]

    print('{:16} {:.3f} {} {} {:.1f} {:6.1f} {} {} {}'.format(*varis))

print('Sample')
var_keys = ['method', 'mean', 'std']
print('{:15} {:15} {:15}'.format(*var_keys))
for key in norm_samples.keys():
    varis = [key, np.mean(np.mean(norm_samples[key], axis=0)), np.mean(np.std(norm_samples[key], axis=0))]
    print('{:15} {:5.13f} {:5.13f}'.format(*varis))

print('Epoch')
print('{:15} {:15} {:15}'.format(*var_keys))
for key in norm_samples.keys():
    varis = [key, np.mean(np.mean(norm_ep[key], axis=0)), np.mean(np.std(norm_ep[key], axis=0))]
    print('{:15} {:5.13f} {:5.13f}'.format(*varis))
print('#################################################')
print('Total errors: {}'.format(err))
print('#################################################')

print(bX.dtype, b_lenX.dtype, maskX.dtype, bY[0].dtype, b_lenY.dtype)
plt.show()
