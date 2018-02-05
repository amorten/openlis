# Defines the DataSet class,
# Used to hold the raw data and feed batches

import numpy as np
import os

class DataSet(object):

  def __init__(self, keys, labels):
    """Construct a DataSet.
    """
   
    assert keys.shape[0] == labels.shape[0], (
          'keys.shape: %s labels.shape: %s' % (keys.shape, labels.shape))

    self._num_keys = keys.shape[0]

    self._keys = np.array(keys)
    self._labels = np.array(labels)
    self._epochs_completed = 0
    self._index_in_epoch = 0

    if len(keys.shape) > 1:
      self._key_size = keys.shape[1]
    else:
      self._key_size = 1

    if len(keys) > 0:
      self._keys_mean = np.mean(keys)
      self._keys_std = np.std(keys)
    else:
      self._keys_mean = None
      self._keys_std = None
      
  @property
  def keys(self):
    return self._keys

  @property
  def labels(self):
    return self._labels

  @property
  def num_keys(self):
    return self._num_keys

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def key_size(self):
    return self._key_size

  @property
  def keys_mean(self):
    return self._keys_mean

  @property
  def keys_std(self):
    return self._keys_std
  
  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""

    
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_keys:
      # Finished epoch
      self._epochs_completed += 1
      if shuffle:
        # Shuffle the data
        perm = np.arange(self._num_keys)
        np.random.shuffle(perm)
        self._keys = self._keys[perm]
        self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_keys
    end = self._index_in_epoch
    return self._keys[start:end], self._labels[start:end]

  def reset_epoch(self):
    self._index_in_epoch = 0
    

def create_train_validate_data_sets(data_set, validation_size=0):
  """Creates training and validation data sets.
  """

  #Shuffle the keys and labels by same permutation
  perm = np.arange(data_set.num_keys)
  np.random.shuffle(perm)
  keys = data_set.keys[perm]
  labels = data_set.labels[perm]

  if not 0 <= validation_size <= len(keys):
    raise ValueError(
        "Validation size should be between 0 and {}. Received: {}."
        .format(len(keys), validation_size))

  validation_keys = keys[:validation_size]
  validation_labels = labels[:validation_size]
  train_keys = keys[validation_size:]
  train_labels = labels[validation_size:]

  train = DataSet(np.reshape(train_keys,[-1,1]), train_labels)
  validation = DataSet(validation_keys, validation_labels)


  class DataSets(object):
    pass

  data_sets = DataSets()
  data_sets.train = train
  data_sets.validate = validation
  return data_sets


def generate_uniform_floats_data_set(num_keys=100000, key_range=[0.0,1.0], iseed=None):
  """Generate a DataSet of uniform floating points.
  """

  np.random.seed(iseed)
  keys = np.random.random(num_keys)
  keys = (key_range[1] - key_range[0]) * keys + key_range[0]
  
  keys = np.sort(keys)
  labels = np.arange(num_keys)

  return DataSet(keys=keys, labels=labels)



def load_keys_npy(dir="./test_data", fname="uniform_floats.npy"):
  """Load keys from .npy file"""

  keys = np.load(os.path.join(dir, fname))
  keys = np.unique(keys) # Unique returns sorted data
  labels = np.arange(len(keys))
  
  return DataSet(keys=keys, labels=labels)


