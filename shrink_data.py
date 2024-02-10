import numpy as np
import h5py
def subsample_data_2018_tofile(from_filename="Data/23mods_61023.hdf5", sample_rate=1/2,to_filename="Data/23mods_1Mdata_61023.hdf5"):
    f = h5py.File(from_filename, 'r')
    X = f['X']  # ndarray(2555904*512*2)
    Y = f['Y']  # ndarray(2M*24)
    Z = f['Z']  # ndarray(2M*1)
    to_file = h5py.File(to_filename,'w')
    n_subsample = int(X.shape[0] * sample_rate)
    X_slice = to_file.create_dataset(shape=(n_subsample,X.shape[1],X.shape[2]),
                                     maxshape = (None,X.shape[1],X.shape[2]),
                                     dtype=np.float32,compression='gzip',name='X',
                                     chunks=(1,X.shape[1],X.shape[2]))
    Y_slice = to_file.create_dataset(shape=(n_subsample,Y.shape[1]),
                                     maxshape=(None,Y.shape[1]),
                                     dtype=np.uint8, compression='gzip', name='Y',
                                     chunks=(1,Y.shape[1]))
    Z_slice = to_file.create_dataset(shape=(n_subsample,1),
                                     dtype=int, compression='gzip', name='Z',
                                     chunks=None)
    snr_count = 26
    batch_size = int(X.shape[0]/snr_count)
    n_slice = int(batch_size * sample_rate)
    # random sample from each snr samples
    for i in range(snr_count):
        print('subsample the snr {}'.format(i))
        batch_X = X[i * batch_size:(i + 1) * batch_size]
        batch_Y = Y[i * batch_size:(i + 1) * batch_size]
        batch_Z = Z[i * batch_size:(i + 1) * batch_size]
        np.random.seed(2016)
        rand_idx = np.random.choice(np.arange(0,batch_size),size=n_slice,replace=False)
        X_slice[i * n_slice:(i + 1) * n_slice, :, :]    = batch_X[rand_idx]
        Y_slice[i * n_slice:(i + 1) * n_slice, :]       = batch_Y[rand_idx]
        Z_slice[i * n_slice:(i + 1) * n_slice]          = batch_Z[rand_idx]
    to_file.close()
    f.close()
    print('subsample complete.total samples:{}'.format(n_subsample))

subsample_data_2018_tofile()

# classes = ['OOK',
#                '4ASK',
#                '8ASK',
#                'BPSK',
#                'QPSK',
#                '8PSK',
#                '16PSK',
#                '32PSK',
#                '16APSK',
#                '32APSK',
#                '64APSK',
#                '128APSK',
#                '16QAM',
#                '32QAM',
#                '64QAM',
#                '128QAM',
#                '256QAM',
#                'AM-SSB-WC',
#                'AM-SSB-SC',
#                'AM-DSB-WC',
#                'AM-DSB-SC',
#                'FM',
#                'GMSK',
#                'OQPSK']
#
# size = 106496
# from_filename ='Data/GOLD_XYZ_OSC.0001_1024.hdf5'
# f = h5py.File(from_filename,'r')  #
# X = f['X'][:,:,:]  # ndarray(159744*1024*2)
# Z = f['Z'][:]  # ndarray(159744*1)
#
# for i, mod in enumerate(classes):
#     to_filename = 'Data/%s.hdf5'%mod
#     to_file = h5py.File(to_filename, 'w')
#
#     X_slice = to_file.create_dataset(shape=(size, X.shape[1], X.shape[2]),
#                                      maxshape=(None, X.shape[1], X.shape[2]),
#                                      dtype=np.float32, compression='gzip', name='X',
#                                      chunks=(1, X.shape[1], X.shape[2]))
#
#
#
#     Z_slice = to_file.create_dataset(shape=(size, 1),
#                                      dtype = int, compression='gzip', name='Z',
#                                      chunks=None)
#
#
#     X_slice[:, :, :] = X[i * size:(i + 1) * size, :, :]
#
#     Z_slice[:] = Z[i * size:(i + 1) * size]
#
#     to_file.close()
#     f.close()
