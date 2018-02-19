# Defines the DataSet class,
# Used to hold the raw data and feed batches

import numpy as np
import os

class DataSet(object):

  def __init__(self, keys, positions=None, num_positions=None):
    """Construct a DataSet.
    """
   
    assert keys.shape[0] == positions.shape[0], (
          'keys.shape: %s positions.shape: %s' % (keys.shape, positions.shape))

    self._num_keys = keys.shape[0]
    
    
    self._keys = np.array(keys)
    if positions is not None:
        self._positions = np.array(positions)
    else:
        self._keys = np.sort(keys)
        self._positions = np.arange(self._num_keys)

    if num_positions is not None:
      self._num_positions = num_positions
    else:
      if len(self._positions) == 0:
        self._num_positions = 0
      else:
        self._num_positions = self._positions[-1] + 1
      
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
  def positions(self):
    return self._positions

  @property
  def num_keys(self):
    return self._num_keys

  @property
  def num_positions(self):
    return self._num_positions

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
        self._positions = self._positions[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_keys
    end = self._index_in_epoch
    return self._keys[start:end], self._positions[start:end]

  def reset_epoch(self):
    self._index_in_epoch = 0
    

def create_train_validate_data_sets(data_set, validation_size=0):
  """Creates training and validation data sets.
  """

  #Shuffle the keys and positions by same permutation
  perm = np.arange(data_set.num_keys)
  np.random.shuffle(perm)
  keys = data_set.keys[perm]
  positions = data_set.positions[perm]

  if not 0 <= validation_size <= len(keys):
    raise ValueError(
        "Validation size should be between 0 and {}. Received: {}."
        .format(len(keys), validation_size))

  validation_keys = keys[:validation_size]
  validation_positions = positions[:validation_size]
  train_keys = keys[validation_size:]
  train_positions = positions[validation_size:]

  train = DataSet(np.reshape(train_keys,[-1,1]), train_positions)
  validation = DataSet(validation_keys, validation_positions)


  class DataSets(object):
    pass

  data_sets = DataSets()
  data_sets.train = train
  data_sets.validate = validation
  return data_sets


def generate_uniform_floats(num_keys=100000, key_range=[0.0,1.0], iseed=None):
  """Generate a DataSet of uniform floating points.
  """

  np.random.seed(iseed)
  keys = np.random.random(num_keys)
  keys = (key_range[1] - key_range[0]) * keys + key_range[0]
  
  keys = np.sort(keys)
  positions = np.arange(num_keys)

  return DataSet(keys=keys, positions=positions)

def generate_normal_floats(num_keys=100000, mean=0, std=1.0, iseed=None):
  """Generate a DataSet of normallaly distributed floating points.
  """

  np.random.seed(iseed)
  keys = np.random.normal(loc=mean, scale=std, size=num_keys)
  
  keys = np.sort(keys)
  positions = np.arange(num_keys)

  return DataSet(keys=keys, positions=positions)



def load_keys_npy(dir="./test_data", fname="uniform_floats.npy"):
  """Load keys from .npy file"""

  keys = np.load(os.path.join(dir, fname))
  keys = np.unique(keys) # Unique returns sorted data
  positions = np.arange(len(keys))
  
  return DataSet(keys=keys, positions=positions)


