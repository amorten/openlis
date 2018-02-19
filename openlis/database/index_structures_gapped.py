
import numpy as np
from ..data import DataSet

class IndexStructureGapped(object):
    """Defines class IndexStructureGapped, which implements Select, Insert,
       and Delete functionality using a numpy array for storage. Gaps
       are left in the array to allow for fast insert operations.
    """
    
    def __init__(self, model, scale):
        """Initialize class

        Args:
            model: An instance of class RMIsimple
            scale: Integer, indicates size of gapped array relative to key array.
            

        Returns:
            -
        """
        
        self._model = model
        self._scale = scale
        
        # Makes a copy of the data_set, and converts to numpy array.
        keys = np.array(model._data_set.keys) 

        if len(keys.shape) > 1:
            if keys.shape[1] > 1:
                raise ValueError("Key_size must be 1")
            else:
                keys = keys.squeeze()

        # Sort the keys, for fast Select, Insert, and Delete
        self._keys_array = np.sort(np.array(keys))

        # Put gaps into the array
        self._is_key = np.full(self._keys_array.shape,True)
        self.rescale()

        
    def rescale(self, scale=None):
        """Rescales the size of the array, adding gaps between keys,
           with the new array being scale times larger.

        Args:
            scale: Integer. New array will be scale times larger.
                   Note that scale must be greater or equal to 1.

        Returns:
            -
            (modifies self.keys_array and self.is_key)
        """

        if scale is not None:
            self._scale = scale
        
        # First remove gaps from array
        self._keys_array = self._keys_array[self._is_key]

        num_keys = self._keys_array.shape[0]
        
        # Construct new array with gaps, using is_key to keep track
        # of which elements are keys.

        trues = np.full(num_keys, True)
        falses =  np.full(num_keys, False)
        keys = self._keys_array

        # Each key should repeat scale number of times
        self._keys_array = np.repeat(keys, self._scale)

        # Initialize _is_key array to False
        self._is_key = np.full(self._scale * num_keys,
                               False)
        # Then mark as true the first instance of each key
        self._is_key[0::self._scale] = True

        

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
            # but must also lie between 0 and max_key_idx
            search_range_left = model_pos - max_error_left
            search_range_left = np.maximum(0,
                                           search_range_left)
            search_range_left = np.minimum(max_key_idx,
                                           search_range_left)

            # Rightmost search pos should typically be (model_pos + max_error_right),
            # but must also lie between 0 and max_key_idx
            search_range_right = model_pos + max_error_right
            search_range_right = np.maximum(0,
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
                If a key is not found, the returned position is -1.
        """
        pos_candidates =  self._find_insert_position(keys)

        num_keys = len(keys)

        pos_output = np.zeros(num_keys,dtype=np.int64)
        
        for idx in range(num_keys):

            key = keys[idx]
            pos = pos_candidates[idx]
            
            if (pos < self._keys_array.shape[0]
                and self._keys_array[pos] == key
                and self._is_key[pos] == True):
                
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
        
        success = np.full(len(keys), False)

        for idx, key in enumerate(keys):        

            pos =  self._find_insert_position([key])
            pos = pos[0]
            
            
            if (pos < self._keys_array.shape[0]
                and self._keys_array[pos] == key
                and self._is_key[pos] == True):
                # If the key already exists, no insertation takes place
                # and the output is success=False
                success[idx] = False
            else:
                # If the key did not already exist, then output success=True.
                success[idx] = True

            # Don't procede with insert if key already exists
            if not success[idx]:
                continue
            
            # Search to the left and right until first available position
            # is found.

            left_pos = pos - 1
            right_pos = pos

            # Check whether there is space to the left or right to search
            if left_pos < 0:
                more_left = False
            else:
                more_left = True
            if right_pos >= self._keys_array.shape[0]:
                more_right = False
            else:
                more_right = True

            # Keep search while there is room to the left or right
            while more_left or more_right:

                if more_right:
                    if self._is_key[right_pos] == False:
                        # If a gap is found, shift the data around and fill in the gap
                        self._is_key[right_pos] = True
                        self._keys_array[pos+1:right_pos+1] = self._keys_array[pos:right_pos]
                        self._keys_array[pos] = key
                        position_range_for_errors_update = [pos, right_pos+1]
                        break
                    else:
                        # If no gap, increment to the right
                        right_pos += 1
                        if right_pos >= self._keys_array.shape[0]:
                            more_right = False

                if more_left:
                    if self._is_key[left_pos] == False:
                        # If a gap is found, shift the data around and fill in the gap
                        self._is_key[left_pos] = True
                        self._keys_array[left_pos:pos-1] = self._keys_array[left_pos+1:pos]
                        self._keys_array[pos - 1] = key
                        position_range_for_errors_update = [left_pos, pos]
                        break
                    else:
                        # If no gap, increment to the left
                        left_pos -= 1
                        if left_pos < 0:
                            more_left = False

            # If the above loop terminates without finding a gap...
            if more_left == False and more_right == False:
                # No insertaion position found
                print("Warning: no gaps left to insert key.")
                success[idx] = False

            # Update errors for all keys that have moved
            pos_range = position_range_for_errors_update
            key_pos = []
            for pos in range(pos_range[0],pos_range[1]):
                if self._is_key[pos]:
                    key_pos.append([[self._keys_array[pos]],
                                    pos])
            self._model.calc_min_max_errors(key_pos)
                    
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
                and self._keys_array[pos] == key
                and self._is_key[pos]):
                # If the key already exists, deletion can take place,
                # so the output is success=False.
                success[idx] = True
            else:
                # If the key did not already exist, then output success=False.
                success[idx] = False

        #self._keys_array = np.delete(self._keys_array, pos_candidates[success])

        self._is_key[pos_candidates[success]] = False
        
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
                default=None (use the model's self.batch_sizes)
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

        
        new_keys = self._keys_array[self._is_key]
        num_keys = new_keys.shape[0]
        new_key_positions = np.arange(self._keys_array.shape[0])[self._is_key]
        data_set = DataSet(keys=np.reshape(new_keys,[-1,1]),
                           positions=new_key_positions,
                           num_positions = self._keys_array.shape[0])

        
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


