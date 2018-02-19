
import numpy as np
from ..data import DataSet

class IndexStructurePacked(object):
    """Defines class IndexStructurePacked, which implements Select, Insert,
       and Delete functionality using a packed numpy array for storage.
       This choice does not allow efficient Insert and Delete, but is 
       straightforward to implement.

       The plan is to implement a new class, IndexStructureGapped, that will
       replace IndexStructurePacked in a future revision. The Gapped version
       will leaves gaps in the array, allowing for faster Insert.

    """
    
    def __init__(self, model):
        """Initialize class

        Args:
            model: An instance of class RMIsimple

        Returns:
            -
        """
        
        self._model = model

        # Makes a copy of the data_set, and converts to numpy array.
        keys = np.array(model._data_set.keys) 

        if len(keys.shape) > 1:
            if keys.shape[1] > 1:
                raise ValueError("Key_size must be 1")
            else:
                keys = keys.squeeze()

        # Sort the keys, for fast Select, Insert, and Delete
        self._keys_array = np.sort(np.array(keys))


    def _find_insert_position(self, keys):
        """Helper function that runs inference to find position
           for Select, Insert, and Delete.

        Args:
            keys: numpy array of keys

        Returns:
            positions: Array of positions. For each key, the returned position
                is leftmost position in the sorted array where Insert would
                result in a sorted array.
        """

        num_keys = len(keys)
        keys = np.reshape(keys, (num_keys, 1))

        # Run inference
        model_positions, experts = self._model.run_inference(keys)
        model_positions = np.reshape(model_positions, (num_keys,))
        experts = np.reshape(experts, (num_keys,))
        
        # Inference prediction is typically not correct position,
        # so we need to conduct a binary search based on known
        # maximum inference error for each expert
        
        pos_output = np.zeros(num_keys,dtype=np.int64)
        for idx in range(num_keys):

            expert = experts[idx]
            model_pos = np.round(model_positions[idx]).astype(np.int64)

            max_error_left = int(self._model.max_error_left[expert])
            max_error_right = int(self._model.max_error_right[expert])

            
            min_pos = self._model.min_pos[expert]
            max_pos = self._model.max_pos[expert]

            max_key_idx = self._keys_array.shape[0] - 1
            
            # Leftmost search pos should typically be (model_pos - max_error_left),
            # but must also lie between 0 and (max_key_idx - max_error_left)
            search_range_left = model_pos - max_error_left
            search_range_left = np.maximum(0,
                                           search_range_left)
            search_range_left = np.minimum(max_key_idx - max_error_left,
                                           search_range_left)

            # Rightmost search pos should typically be (model_pos + max_error_right),
            # but must also lie between max_error_right and max_key_idx
            search_range_right = model_pos + max_error_right
            search_range_right = np.maximum(max_error_right,
                                            search_range_right)
            search_range_right = np.minimum(max_key_idx,
                                            search_range_right)
            
            
            search_range = [search_range_left, search_range_right]

            # Before conducting the search, check whether the error bounds are large enough
            leftmost_key = self._keys_array[search_range[0]]
            rightmost_key = self._keys_array[search_range[1]]

            if leftmost_key <= keys[idx] <= rightmost_key:
                # If the key lies within the range, search for it with binary search
                found_pos = np.searchsorted(self._keys_array[search_range[0]:search_range[1]+1],
                                            keys[idx],
                                            side='left')
                # Because np.searchsorted returns an array with one element:
                found_pos = found_pos[0]
                # Adjust found_pos for full keys_array, not just for the slice
                found_pos += search_range[0]
                
            elif leftmost_key > keys[idx]:
                # If the key lies to the left of the range, just scan to the left incrementally
                pos = search_range[0] - 1
                while pos >= 0:
                    if self._keys_array[pos] < keys[idx]:
                        found_pos = pos + 1
                        break
                    pos -= 1
                if pos == -1:
                    found_pos = 0
                
            elif rightmost_key < keys[idx]:
                # If the key lies to the right of the range, just scan to the right incrementally
                pos = search_range[1] + 1
                while pos <= self._keys_array.shape[0] - 1:
                    if self._keys_array[pos] >= keys[idx]:
                        found_pos = pos
                        break
                    pos += 1
                if pos == self._keys_array.shape[0]:
                    found_pos = pos

            pos_output[idx] = found_pos
                                        
        return pos_output

    
    def select(self, keys):
        """Return position(s) of key(s) in sorted array.

        Args:
            keys: Numpy array of keys.

        Returns:
            positions: Numpy array positions of the keys.
                If a key is not found, its position is set to -1.
        """
        pos_candidates =  self._find_insert_position(keys)

        num_keys = len(keys)

        pos_output = np.zeros(num_keys,dtype=np.int64)
        
        for idx in range(num_keys):

            key = keys[idx]
            pos = pos_candidates[idx]
            
            if (pos < self._keys_array.shape[0]
                and self._keys_array[pos] == key):
                
                pos_output[idx] = pos_candidates[idx]
            else:
                pos_output[idx] = -1

        return pos_output
        
        
        

    def insert(self,keys):
        """Insert key(s) in sorted array.

        Args:
            keys: Numpy array of keys.

        Returns:
            success: Numpy boolean array: True for each successful insertion;
                False for each failed insertion (due to key already in array).
        """

        pos_candidates =  self._find_insert_position(keys)
        
        num_keys = len(keys)

        success = np.full(num_keys, False)
        
        for idx in range(num_keys):

            key = keys[idx]
            pos = pos_candidates[idx]
            
            if (pos < self._keys_array.shape[0]
                and self._keys_array[pos] == key):
                # If the key already exists, no insertation takes place
                # and the output is success=False
                success[idx] = False
            else:
                # If the key did not already exist, then output success=True.
                success[idx] = True

        # Only work with keys that are to be inserted
        keys = np.array(keys)[success]
        pos = np.array(pos_candidates)[success]
        
        # When inserting multiple keys, divide the keys into three groups:
        # keys_before: keys to be inserted to left of self._keys_array
        # keys_middle: keys to be inserted within the array
        # keys_after: keys to be inserted to right of self._keys_array

        # Sort by key
        perm = np.argsort(keys)
        keys = keys[perm]
        pos = pos[perm]

        keys_array_size = self._keys_array.shape[0]

        left_idx = np.searchsorted(pos,0,side='right')
        right_idx = np.searchsorted(pos,keys_array_size,side='left')

        #keys_before = keys[:left_idx]
        #keys_middle = keys[left_idx:right_idx]
        #keys_after = keys[right_idx:]

        # Alternatively, use np.split to get list of three arrays
        # described above.
        keys_split = np.split(keys,[left_idx,right_idx])
        pos_split = np.split(pos,[left_idx,right_idx])
        
        # Insert into the middle first:
        # Newer version of numpy allow multiple inserts
        self._keys_array = np.insert(self._keys_array,
                                     pos_split[1],
                                     keys_split[1])

        # Now concatenate with keys_before and keys_after
        self._keys_array = np.concatenate((keys_split[0],
                                           self._keys_array,
                                           keys_split[2]))

        return success
    

    def delete(self,keys):
        """Delete key(s) in sorted array.

        Args:
            keys: Numpy array of keys.

        Returns:
            success: Numpy boolean array: True for each successful deletion;
                False for each failed deletion (due to key not in array).
        """
        
        pos_candidates =  self._find_insert_position(keys)

        num_keys = len(keys)

        success = np.full(num_keys, False)
        
        for idx in range(num_keys):

            key = keys[idx]
            pos = pos_candidates[idx]
            
            if (pos < self._keys_array.shape[0]
                and self._keys_array[pos] == key):
                # If the key already exists, deletion can take place,
                # so the output is success=False.
                success[idx] = True
            else:
                # If the key did not already exist, then output success=False.
                success[idx] = False

        self._keys_array = np.delete(self._keys_array, pos_candidates[success])
                
        return success

    
    def train(self,
              batch_sizes=None,
              max_steps=None,
              learning_rates=None,
              model_save_dir=None):
        """Train the model, calculate expert errors, etc. Fully prepares
           the model for Select, Insert, and Delete operations.
           This function should be used after significant number of
           insertions and deletions.

        Args:
            batch_sizes: list (length=2) of batch sizes for the two Stages.
                default=None (use the model's  self.batch_sizes)
            max_steps: list (length=2) of number of training steps for the two Stages.
                default=None (use the model's self.max_steps)
            learning_rates: list (length=2) of learning rates for the two Stages.
                default=None (use the model's self.learning_rates)
            model_save_dir: Name of directory to save the model
                default=None (use the model's self.model_save_dir)

        Returns:
            -
        """
        
        # Construct new DataSet from current array of keys,
        # to be used to retrain the model.
        # Assumes that self._keys_array is already sorted.

        num_keys = self._keys_array.shape[0]
        data_set = DataSet(np.reshape(self._keys_array,[-1,1]), np.arange(num_keys))

        
        self._model.new_data(data_set)

        # Train on the new data set.
        self._model.run_training(batch_sizes=batch_sizes,
                                 max_steps=max_steps,
                                 learning_rates=learning_rates,
                                 model_save_dir=model_save_dir)
        
        # Prepare weights for fast inference
        self._model.get_weights_from_trained_model()

        # Calculate the Stage 2 errors
        self._model.calc_min_max_errors()

        
