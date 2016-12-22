import os
import h5py
import scipy.io as sio
from hmmlearn import hmm
import numpy as np
import joblib


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
                                                        maxshape=(None, n_inner_column), chunks=True, dtype='float64')

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


def calculate_statistical_descriptors(h5py_object, filename, dataset):

    # sliding window properties
    window_size = 60
    step = 1
    chunks = sliding_window(dataset, window_size, step)

    # mean, variance and label lists
    mean_list = list()
    variance_list = list()
    min_list = list()
    max_list = list()

    for segmented_data in chunks:
        # separate the labels from the dataset
        n_dataset = segmented_data[:, :]

        # calculate statistical descriptors
        mean = np.mean(a=n_dataset, axis=0)
        var = np.var(a=n_dataset, axis=0)
        mn = np.min(a=n_dataset, axis=0)
        mx = np.max(a=n_dataset, axis=0)

        mean_list.append(mean)
        variance_list.append(var)
        min_list.append(mn)
        max_list.append(mx)

    # list converted to numpy arrays for future processing
    mean_points = np.array(mean_list)
    var_points = np.array(variance_list)
    min_points = np.array(min_list)
    max_points = np.array(max_list)

    # create a new array with all the points appended as new columns
    statistical_descriptors = np.c_[mean_points, min_points, max_points, var_points]
    # standardization : transfer data to have zero mean and unit variance
    sd_mean = np.mean(a=statistical_descriptors, axis=0)
    sd_std = np.std(a=statistical_descriptors, axis=0)
    n_statistical_descriptors = (statistical_descriptors - sd_mean) / sd_std

    # the name of the dataset is the name of the file being process
    # the data of the dataset consist of the actual data for columns 0,...,n-1 and the labels in the last column
    h5py_object.create_dataset(name=filename, data=n_statistical_descriptors, dtype='float32')


def sliding_window(sequence, window_size, step=1):
    """
    Returns a generator that will iterate through
    the defined chunks of input sequence. Input sequence
    must be sliceable.
    """

    # Verify the inputs
    if not isinstance(type(window_size), type(0)) and isinstance(type(step), type(0)):
        raise Exception("**ERROR** type(window_size) and type(step) must be int.")
    if step > window_size:
        raise Exception("**ERROR** step must not be larger than window_size.")
    if window_size > len(sequence):
        raise Exception("**ERROR** window_size must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    number_of_chunks = ((len(sequence) - window_size) / step) + 1

    # Do the work
    for i in range(0, number_of_chunks * step, step):
        yield sequence[i: i + window_size]


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
    h5py_object.create_dataset(name=n_filename, data=values, dtype='float64')


def convert_matlab_h5py(program_path):
    """
    Creates h5py files from all the matlab files
    :param program_path: matlab folder directory
    :return: None
    """
    dataset_path = os.path.join(program_path, 'Dog_5')

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


def process_data_points(program_path):
    """
    calculate statistical descriptors of the datasets
    :param program_path: location of the dataset
    :return: stores values calculated into another file starting with 'processed_'
    """
    dataset_path = os.path.join(program_path, 'dataset')

    # files inside the folder
    h5py_files = next(os.walk(dataset_path))[2]

    writing_file_path = os.path.join(dataset_path, 'processed_interictal_training_dataset.hdf5')
    interictal_writing_object = h5py.File(name=writing_file_path, mode='w')
    writing_file_path = os.path.join(dataset_path, 'processed_preictal_training_dataset.hdf5')
    preictal_writing_object = h5py.File(name=writing_file_path, mode='w')
    writing_file_path = os.path.join(dataset_path, 'processed_testing_training_dataset.hdf5')
    testing_writing_object = h5py.File(name=writing_file_path, mode='w')

    for s_file in h5py_files:
        if 'interictal_dataset' in s_file:
            print 'processing interictal'
            file_path = os.path.join(dataset_path, s_file)
            interictal_object = h5py.File(file_path, 'r')
            # loop through all the keys of interictal
            total_number_keys = len(interictal_object.keys())
            for index_key, key in enumerate(interictal_object.keys()):
                print 'processing key={0}, {1} out of {2}'.format(key, index_key, total_number_keys)
                calculate_statistical_descriptors(h5py_object=interictal_writing_object,
                                                  filename=key,
                                                  dataset=interictal_object[key].value.transpose())
            interictal_object.close()
        if 'preictal_dataset' in s_file:
            print 'processing preictal'
            file_path = os.path.join(dataset_path, s_file)
            preictal_object = h5py.File(file_path, 'r')
            total_number_keys = len(preictal_object.keys())
            for index_key, key in enumerate(preictal_object.keys()):
                print 'processing key={0}, {1} out of {2}'.format(key, index_key, total_number_keys)
                calculate_statistical_descriptors(h5py_object=preictal_writing_object,
                                                  filename=key,
                                                  dataset=preictal_object[key].value.transpose())
            preictal_object.close()
        if 'testing_dataset' in s_file:
            print 'processing testing'
            file_path = os.path.join(dataset_path, s_file)
            testing_object = h5py.File(file_path, 'r')
            total_number_keys = len(testing_object.keys())
            for index_key, key in enumerate(testing_object.keys()):
                print 'processing key={0}, {1} out of {2}'.format(key, index_key, total_number_keys)
                calculate_statistical_descriptors(h5py_object=testing_writing_object,
                                                  filename=key,
                                                  dataset=testing_object[key].value.transpose())
            testing_object.close()

    interictal_writing_object.close()
    preictal_writing_object.close()
    testing_writing_object.close()


def concatenate_data_points(program_path):
    """
    Compress all the h5py files into a training dataset
    :param program_path: location of the h5py files
    :return: training and testing h5py objects
    """

    dataset_path = os.path.join(program_path, 'dataset')

    # files inside the folder
    h5py_files = next(os.walk(dataset_path))[2]

    for s_file in h5py_files:
    #     if 'processed_interictal' in s_file:
    #         interictal_class = Base(input_path=dataset_path, filename='final_interictal_training_dataset')
    #         print 'processing interictal'
    #         file_path = os.path.join(dataset_path, s_file)
    #         interictal_object = h5py.File(file_path, 'r')
    #         # loop through all the keys of interictal
    #         total_number_keys = len(interictal_object.keys())
    #         for index_key, key in enumerate(interictal_object.keys()):
    #             print 'processing key={0}, {1} out of {2}'.format(key, index_key, total_number_keys)
    #             interictal_class.add_dataset(dataset=interictal_object[key].value, labels=1)
    #         interictal_object.close()
        if 'processed_preictal' in s_file:
            preictal_class = Base(input_path=dataset_path, filename='final_preictal_training_dataset')
            print 'processing preictal'
            file_path = os.path.join(dataset_path, s_file)
            preictal_object = h5py.File(file_path, 'r')
            total_number_keys = len(preictal_object.keys())
            for index_key, key in enumerate(preictal_object.keys()):
                print 'processing key={0}, {1} out of {2}'.format(key, index_key, total_number_keys)
                preictal_class.add_dataset(dataset=preictal_object[key].value, labels=0)
            preictal_object.close()
    #     if 'processed_testing' in s_file:
    #         testing_class = Base(input_path=dataset_path, filename='final_testing_pre-inter_ictal_dataset')
    #         print 'processing testing'
    #         file_path = os.path.join(dataset_path, s_file)
    #         testing_object = h5py.File(file_path, 'r')
    #         total_number_keys = len(testing_object.keys())
    #         for index_key, key in enumerate(testing_object.keys()):
    #             print 'processing key={0}, {1} out of {2}'.format(key, index_key, total_number_keys)
    #             testing_class.add_dataset(dataset=testing_object[key].value, labels=0)
    #         testing_object.close()


def hmm_build_train(program_path):

    dataset_path = os.path.join(program_path, 'dataset')

    print 'creating the datasets path'
    preictal_data_path = os.path.join(dataset_path, 'final_preictal_training_dataset.hdf5')
    interictal_data_path = os.path.join(dataset_path, 'final_interictal_training_dataset.hdf5')
    testing_data_path = os.path.join(dataset_path, 'processed_testing_training_dataset.hdf5')

    preictal_model_loaded = False
    interictal_model_loaded = False

    models_path = os.path.join(program_path, 'models')
    # check if model are saved
    if os.path.exists(models_path):
        # hmm inside the models' folder
        hmm_files = next(os.walk(models_path))[2]

        for m_file in hmm_files:
            if ('hmm_preictal' in m_file) and ('.npy' not in m_file):
                # calculate the whole path
                data_path = os.path.join(models_path, m_file)
                # load the model
                preictal_hmm = joblib.load(data_path)
                # turn on flag so the code does not re-train the model
                preictal_model_loaded = True
            elif ('hmm_interictal' in m_file) and ('.npy' not in m_file):
                # calculate the whole path
                data_path = os.path.join(models_path, m_file)
                # load the model
                interictal_hmm = joblib.load(data_path)
                # turn on flag so the code does not re-train the model
                interictal_model_loaded = True

    # create location for storing models for later use
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    # check if model loaded
    if not preictal_model_loaded:
        print 'loading preictal dataset'
        preictal_dataset = h5py.File(name=preictal_data_path, mode='r')
        # calculate the length of each of the unique matlab files conforming the preictal dataset
        list_of_lengths = [239766] * 29
        rest_of_array = int(preictal_dataset['training data'].shape[0] - np.sum(list_of_lengths))
        list_of_lengths.append(rest_of_array)
        preictal_length = np.array(list_of_lengths)
        
        if np.sum(preictal_length) != preictal_dataset['training data'].shape[0]:
            raise ValueError('preictal length variable does not match preictal dataset length')
        print 'creating a preictal Gaussian HMM object'
        preictal_hmm = hmm.GaussianHMM(n_components=8, verbose=True)
        print '\ttraining the model'
        preictal_hmm.fit(preictal_dataset['training data'], preictal_length)

        print '\tstoring model'
        hmm_preictal_path_filename = os.path.join(models_path, 'hmm_preictal')
        joblib.dump(preictal_hmm, hmm_preictal_path_filename)

    # check if model loaded
    if not interictal_model_loaded:
        print 'loading interictal dataset'
        interictal_dataset = h5py.File(name=interictal_data_path, mode='r')
        # calculate the length of each of the unique matlab files conforming the interictal dataset
        list_of_lengths = [239766] * 449
        rest_of_array = int(interictal_dataset['training data'].shape[0] - np.sum(list_of_lengths))
        list_of_lengths.append(rest_of_array)
        interictal_length = np.array(list_of_lengths)

        if np.sum(interictal_length) != interictal_dataset['training data'].shape[0]:
            raise ValueError('preictal length variable does not match preictal dataset length')
        # interictal_length = np.array([239766] * 300)
        # interictal_length = np.array([239766] * 200)
        # interictal_length = np.array([239766] * 100)
        print 'creating a interictal Gaussian HMM object'
        interictal_hmm = hmm.GaussianHMM(n_components=8, verbose=True)
        print '\ttraining the model'
        # 450
        interictal_hmm.fit(interictal_dataset['training data'], interictal_length)
        # 200
        # interictal_hmm.fit(interictal_dataset['training data'][:47953200], interictal_length)
        # 100
        # interictal_hmm.fit(interictal_dataset['training data'][:23976600], interictal_length)

        print '\tstoring model'
        hmm_interictal_path_filename = os.path.join(models_path, 'hmm_interictal')
        joblib.dump(preictal_hmm, hmm_interictal_path_filename)

    print 'loading testing dataset'
    testing_dataset = h5py.File(name=testing_data_path, mode='r')

    true_results = obtain_true_results()
    true_count = 0.0

    output_file = open('results.csv','w')
    for testing_key in testing_dataset.keys():
        print 'calculating likelihoodi for {0}'.format(testing_key)
        interictal_log_prob, _ = interictal_hmm.decode(testing_dataset[testing_key].value)
        preictal_log_prob, _ = preictal_hmm.decode(testing_dataset[testing_key].value)

        if interictal_log_prob > preictal_log_prob:
            # 0 = interictal
            if true_results[testing_key] == 0:
                true_count += 1
            row_w = '{0},{1}'.format(testing_key,0)
            output_file.write(row_w)
            output_file.write('\n')
        else:
            # 1 = preictal
            if true_results[testing_key] == 1:
                true_count += 1
            row_w = '{0},{1}'.format(testing_key,1)
            output_file.write(row_w)
            output_file.write('\n')

    accuracy = true_count / len(testing_dataset.keys())
    print 'accuracy={0}'.format(accuracy)


def obtain_true_results():
    """
    :return: the ground true labels in a dictionary variable
    """

    true_dict = dict()
    true_predictions = open('SzPrediction_answer_key.csv')
    for line in true_predictions:
        line_separated = line.split(',')
        key_file = line_separated[0].replace('.mat', '')
        prediction_value = line_separated[1]
        true_dict[key_file] = int(prediction_value)

    return true_dict

if __name__ == '__main__':

    # big-data project path
    current_program_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])

    convert_matlab_h5py(program_path=current_program_path)
    process_data_points(program_path=current_program_path)
    concatenate_data_points(program_path=current_program_path)
    hmm_build_train(program_path=current_program_path)
