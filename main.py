import os
import h5py
import scipy.io as sio
from hmmlearn import hmm
import numpy as np


class Base:
    def __init__(self, input_path, filename):
        # location of the file
        self.input_path = input_path
        # flag used when adding files
        self.first_file = True
        # index to keep track of size of dataset
        self.last_index = 0
        # length list used for HMM training
        self.training_dataset_lengths = list()
        # name of the file
        self.filename = filename + '.hdf5'
        # new training dataset location
        self.training_path_filename = ''
        # h5py object where values are store
        self.training_dataset_object = ''

        # temporary file to store training and testing data information
        self.training_path_filename = os.path.join(input_path, self.filename)
        self.training_dataset_object = h5py.File(self.training_path_filename, 'w')

    def add_dataset(self, dataset, labels):

        # shape of the inner dataset
        n_inner_row, n_inner_column = dataset.shape

        # get the size of the new dataset
        total_rows = self.last_index + n_inner_row

        if self.first_file:
            self.training_dataset_object.create_dataset(name='training data',
                                                        shape=(n_inner_row, n_inner_column),
                                                        maxshape=(None, n_inner_column), chunks=True)

            self.training_dataset_object['training data'][:, :] = dataset
            self.first_file = False

        else:

            # resize the dataset to accommodate the new data
            self.training_dataset_object['training data'].resize(total_rows, axis=0)
            # add new data
            self.training_dataset_object['training data'][
                self.last_index:] = dataset

        # increase the dataset size
        self.last_index = total_rows
        # add dataset size to a list of lengths
        self.training_dataset_lengths.append(n_inner_row)

    def get_values(self, dataset_name):
        self.training_dataset_object[dataset_name]


def add_dataset(h5py_object, values, filename):
    """
    Add the points to the h5py variable
    :param h5py_object: h5py object dataset
    :param filename: name of the file to add
    :param values: data points
    :return: None
    """
    # removing .mat file format
    n_filename = filename.replace('.mat', '')
    # add dataset to object
    h5py_object.create_dataset(name=n_filename, data=values)


def convert_matlab_h5py(dataset_path):
    """
    Creates h5py files from all the matlab files
    :param dataset_path: matlab folder directory
    :return: None
    """
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


def concatenate_data_points(dataset_path):
    """
    Compress all the h5py files into a training dataset
    :param dataset_path: location of the h5py files
    :return: training and testing h5py objects
    """

    # files inside the folder
    h5py_files = next(os.walk(dataset_path))[2]

    for s_file in h5py_files:
        if 'interictal_dataset' in s_file:
            interictal_class = Base(input_path=dataset_path, filename='interictal_training_dataset')
            print 'processing interictal'
            file_path = os.path.join(dataset_path, s_file)
            interictal_object = h5py.File(file_path, 'r')
            # loop through all the keys of interictal
            total_number_keys = len(interictal_object.keys())
            for index_key, key in enumerate(interictal_object.keys()):
                print 'processing key={0}, {1} out of {2}'.format(key, index_key, total_number_keys)
                interictal_class.add_dataset(dataset=interictal_object[key].value.transpose(), labels=1)
            interictal_object.close()
        elif 'preictal_dataset' in s_file:
            preictal_class = Base(input_path=dataset_path, filename='preictal_training_dataset')
            print 'processing preictal'
            file_path = os.path.join(dataset_path, s_file)
            preictal_object = h5py.File(file_path, 'r')
            total_number_keys = len(preictal_object.keys())
            for index_key, key in enumerate(preictal_object.keys()):
                print 'processing key={0}, {1} out of {2}'.format(key, index_key, total_number_keys)
                preictal_class.add_dataset(dataset=preictal_object[key].value.transpose(), labels=0)
            preictal_object.close()
        elif 'testing_dataset' in s_file:
            testing_class = Base(input_path=dataset_path, filename='testing_pre-inter_ictal_dataset')
            print 'processing testing'
            file_path = os.path.join(dataset_path, s_file)
            testing_object = h5py.File(file_path, 'r')
            total_number_keys = len(testing_object.keys())
            for index_key, key in enumerate(testing_object.keys()):
                print 'processing key={0}, {1} out of {2}'.format(key, index_key, total_number_keys)
                testing_class.add_dataset(dataset=testing_object[key].value.transpose(), labels=0)
            testing_object.close()


def hmm_build_train(dataset_path):

    print 'creating the datasets path'
    preictal_data_path = os.path.join(dataset_path, 'preictal_training_dataset.hdf5')
    interictal_data_path = os.path.join(dataset_path, 'interictal_training_dataset.hdf5')
    testing_data_path = os.path.join(dataset_path, 'testing_dataset.hdf5')

    print 'calculating the length of each of the unique matlab files conforming the testing dataset'
    testing_length = np.array([239766] * 191)

    print 'loading the dataset'
    preictal_dataset = h5py.File(name=preictal_data_path, mode='r')
    interictal_dataset = h5py.File(name=interictal_data_path, mode='r')
    testing_dataset = h5py.File(name=testing_data_path, mode='r')

    # calculate the length of each of the unique matlab files conforming the preictal dataset
    preictal_length = np.array([239766] * 30)
    print 'creating a preictal Gaussian HMM object'
    preictal_hmm = hmm.GaussianHMM(n_components=8, verbose=True)
    print '\ttraining the model'
    preictal_hmm.fit(preictal_dataset['training data'], preictal_length)

    # calculate the length of each of the unique matlab files conforming the interictal dataset
    # interictal_length = np.array([239766] * 450)
    interictal_length = np.array([239766] * 30)
    print 'creating a interictal Gaussian HMM object'
    interictal_hmm = hmm.GaussianHMM(n_components=8, verbose=True)
    print '\ttraining the model'
    interictal_hmm.fit(interictal_dataset['training data'][:7192980], interictal_length)

    for testing_key in testing_dataset.keys():
        print 'calculating likelihood'
        interictal_log_prob, _ = interictal_hmm.decode(testing_data_path[testing_key], testing_length)
        preictal_log_prob, _ = preictal_hmm.decode(testing_data_path[testing_key], testing_length)

        if interictal_log_prob > preictal_log_prob:
            # 0 = interictal
            print 'data= {0} prediction {1}'.format(testing_key, 0)
        else:
            # 1 = preictal
            print 'data= {0} prediction {1}'.format(testing_key, 1)

if __name__ == '__main__':

    # big-data project path
    program_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    dataset_dir_path = os.path.join(program_path, 'dataset')

    # convert_matlab_h5py(dataset_path='/Users/jguerra/PycharmProjects/Big-Data-Project/Dog_5')
    # concatenate_data_points(dataset_path='/Users/jguerra/PycharmProjects/Big-Data-Project/dataset')
    hmm_build_train(dataset_path=dataset_dir_path)




