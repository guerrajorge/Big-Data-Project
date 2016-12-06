import os
import h5py
import scipy.io as sio


def add_dataset(h5py_object, filename, values):
    # removing .mat file format
    n_filename = filename.replace('.mat', '')
    # add dataset to object
    h5py_object.create_dataset(name=n_filename, data=values)


def convert_matlab_h5py(dataset_path):
    # files inside the folder
    matlab_files = next(os.walk(dataset_path))[2]

    preictal_dataset = h5py.File('preictal_dataset.hdf5', 'w')
    interictal_dataset = h5py.File('interictal_dataset.hdf5', 'w')
    testing_dataset = h5py.File('testing_dataset.hdf5', 'w')

    # statistics
    n_preictal = 0
    n_interictal = 0
    n_test = 0

    for matlab_file in matlab_files:
        # load matlab file
        file_path = os.path.join(dataset_path, matlab_file)

        # make sure its a matlab file
        if '.mat' in file_path:
            matlab_content = sio.loadmat(file_path)
            # every matlab file has an specific variable name to access the data
            possible_keys = matlab_content.keys()
            matlab_key = ''

            # flag to know files being read
            interictal_flag = False
            preictal_flag = False
            test_flag = False

            # get the inter-ictal. pre-ictal or test variable name
            for included_keys in possible_keys:
                if 'interictal' in included_keys:
                    interictal_flag = True
                    matlab_key = included_keys
                    n_interictal += 1
                    break
                if 'preictal' in included_keys:
                    preictal_flag = True
                    matlab_key = included_keys
                    n_preictal += 1
                    break
                if 'test' in included_keys:
                    test_flag = True
                    matlab_key = included_keys
                    n_test += 1
                    break

            # location of the data
            values = matlab_content[matlab_key]['data'][0][0]

            print 'adding filename={0}'.format(matlab_file)

            if preictal_flag:
                add_dataset(h5py_object=preictal_dataset, filename=matlab_file, values=values)
            elif interictal_flag:
                add_dataset(h5py_object=interictal_dataset, filename=matlab_file, values=values)
            elif test_flag:
                add_dataset(h5py_object=testing_dataset, filename=matlab_file, values=values)
            else:
                raise ValueError('Wrong matlab filename {0}'.format(matlab_file))

    print 'finished adding matlab files'
    print 'number preictal={0}\nnumber interictal={1}\nnumber test={2}'.format(n_preictal, n_interictal, n_test)

if __name__ == '__main__':

    training_data = convert_matlab_h5py(dataset_path='/Users/jguerra/PycharmProjects/Big-Data-Project/Dog_5')



