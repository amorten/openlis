
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import os
import math
import numpy as np
import tensorflow as tf
from six.moves import xrange



class RMI_simple(object):
    """ Implements the simple "Recursive-index model" described in the paper
        'The Case for Learned Index Structures', which can be found at
        [Kraska et al., 2017](http://arxiv.org/abs/1712.01208)
        ([pdf](http://arxiv.org/pdf/1712.01208.pdf)).

        The first stage is a fully connected neural network with any number
        (>=0) of hidden layers. Each second stage model is a single-variable
        linear regression.

        At model creation, the user can choose the widths of the
        hidden layers and the number of models ("experts") used in
        stage 2.
    """

    def __init__(self,
                 data_set,
                 hidden_layer_widths=[16,16],
                 num_experts=10,
                 learning_rates = [0.1,0.1],
                 max_steps = [1000,1000],
                 batch_sizes = [1000,1000],
                 model_save_dir = 'tf_checkpoints'):
        """Initializes the Recursive-index model

        Args:
            data_set: object of type DataSet, which the model will train on
            hidden layer_widths: list of hidden layer widths (use empty list
                                 for zero hidden layers)
            num_experts: number of models ("experts") used in stage 2
            learning_rates: list (length=2) of learning rates for each stage
            max_steps: list (length=2) of maximum number of training steps for each stage
            batch_sizes: list (length=2) of batch training sizes for each stage
            model_save_dir: Name of directory to save model
            
        """

        # Initialize from input parameters
        self._data_set = data_set    
        self.hidden_layer_widths = hidden_layer_widths
        self.num_experts = num_experts
        self.learning_rates = learning_rates
        self.max_steps = max_steps
        self.batch_sizes = batch_sizes
        self.model_save_dir = model_save_dir

        # Decide which optimized inference function to use, based on
        # number of hidden layers.

        num_hidden_layers = len(self.hidden_layer_widths)
        
        if num_hidden_layers == 0:
            self.run_inference = self._run_inference_numpy_0_hidden
        elif num_hidden_layers == 1:
            self.run_inference = self._run_inference_numpy_1_hidden
        elif num_hidden_layers == 2:
            self.run_inference = self._run_inference_numpy_2_hidden
        else:
            self.run_inference = self._run_inference_numpy_n_hidden
            
        # Store prediction errors for each expert
        # Fill these values using self.calc_min_max_errors()
        self.max_error_left = None
        self.max_error_right = None        
        self.min_predict = None
        self.max_predict = None
        self.min_pos = None
        self.max_pos = None

        self._initialize_errors()

        # Define variables to stored trained tensor variables
        # (e.g. weights and biases).
        # These are used to run inference faster with numpy
        # rather than with TensorFlow.

        self.hidden_w = [None] * num_hidden_layers
        self.hidden_b = [None] * num_hidden_layers
        self.linear_w = None
        self.linear_b = None
        self.stage_2_w = None
        self.stage_2_b = None
        self._expert_factor = None

        # Pre-calculate some normalization and computation constants,
        # so that they are not repeatedly calculated later.

        # Normalize using mean and dividing by the standard deviation
        self._keys_mean = self._data_set.keys_mean
        self._keys_std_inverse = 1.0 / self._data_set.keys_std
        # Normalize further by dividing by 2*sqrt(3), so that
        # a uniform distribution in the range [a,b] would transform
        # to a uniform distribution in the range [-0.5,0.5]
        self._keys_norm_factor = 0.5 / np.sqrt(3)
        # Precalculation for expert = floor(stage_1_pos * expert_factor)
        self._expert_factor = self.num_experts/self._data_set.num_positions

        
    def new_data(self, data_set):
        """Changes the data set used for training. For example, this function should
           be called after a large number of inserts are performed.

        Args:
            data_set: type DataSet, replaces current data_set with new data_set
        """
        
        self._data_set = data_set

        # Normalize using mean and dividing by the standard deviation
        self._keys_mean = self._data_set.keys_mean
        self._keys_std_inverse = 1.0 / self._data_set.keys_std
        # Normalize further by dividing by 2*sqrt(3), so that
        # a uniform distribution in the range [a,b] would transform
        # to a uniform distribution in the range [-0.5,0.5]
        self._keys_norm_factor = 0.5 / np.sqrt(3)
        # Precalculation for expert = floor(stage_1_pos * expert_factor)
        self._expert_factor = self.num_experts/self._data_set.num_positions

        
    def _setup_placeholder_inputs(self,batch_size):
        """Create placeholder tensors for inputing keys and positions.

        Args:
            batch_size: Batch size.

        Returns:
            keys_placeholder: Keys placeholder tensor.
            labels_placeholder: Labels placeholder tensor.
        """

        # The first dimension is None for both placeholders in order
        # to handle variable batch sizes
        with tf.name_scope("placeholders"):
            keys_placeholder = tf.placeholder(tf.float32, shape=(None,self._data_set.key_size), name="keys")
            labels_placeholder = tf.placeholder(tf.int64, shape=(None), name="labels")
        return keys_placeholder, labels_placeholder


    def _fill_feed_dict(self, keys_pl, labels_pl, batch_size=100, shuffle=True):
        """ Creates a dictionary for use with TensorFlow's feed_dict

        Args:
            keys_pl: TensorFlow (TF) placeholder for keys,
                     created from self._setup_placeholder_inputs().
            labels_pl: TF placeholder for labels (i.e. the key positions)
                     created from self._setup_placeholder_inputs().
            batch_size: integer size of batch
            shuffle: whether or not to shuffle the data
                     Note: shuffle=Flase can be useful for debugging

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """

        # Create the feed_dict for the placeholders filled with the next
        # `batch size` examples.
        
        keys_feed, labels_feed = self._data_set.next_batch(batch_size,shuffle)
        feed_dict = {
            keys_pl: keys_feed,
            labels_pl: labels_feed,
        }
        return feed_dict

    
    def _setup_inference_stage_1(self, keys):
        """Set up Stage 1 inference.

        Args:
            keys: Tensorflow placeholder for keys

        Returns:
            pos_stage_1: Output tensor that predicts key position

        """

        # All Stage 1 operations should be in 'stage_1' name_Scope
        with tf.name_scope('stage_1'):

            keys_std = self._data_set.keys_std
            keys_mean = self._data_set.keys_mean
            key_size = self._data_set.key_size

            hidden_widths = self.hidden_layer_widths
            
            # Normalize
            with tf.name_scope('normalize'):

                keys = tf.cast(keys,dtype=tf.float64)
                
                # Normalize using mean and standard deviation
                keys_normed = tf.scalar_mul(tf.constant(1.0/keys_std),
                                            tf.subtract(keys,tf.constant(keys_mean)))

                # Normalize further by dividing by 2*sqrt(3), so that
                # a uniform distribution in the range [a,b] would transform
                # to a uniform distribution in the range [-0.5,0.5]

                keys_normed = tf.scalar_mul(tf.constant(0.5/np.sqrt(3)),
                                            keys_normed)
            
            # All hidden layers
            tf_output = keys_normed # previous output
            output_size = key_size # previous output size
            for layer_idx in range(0,len(hidden_widths)):
                tf_input = tf_output # get current inputs from previous outputs
                input_size = output_size
                output_size = hidden_widths[layer_idx]
                name_scope = "hidden_" + str(layer_idx+1) # Layer num starts at 1
                with tf.name_scope(name_scope):
                    weights = tf.Variable(
                        tf.truncated_normal([input_size, output_size],
                                            stddev=1.0 / math.sqrt(float(input_size)),
                                            dtype=tf.float64),
                        name='weights',
                        dtype=tf.float64)
                    biases = tf.Variable(tf.zeros([output_size],dtype=tf.float64),
                                         name='biases',
                                         dtype=tf.float64)
                    tf_output = tf.nn.relu(tf.matmul(tf_input, weights) + biases)

                
            # Linear
            with tf.name_scope('linear'):
                weights = tf.Variable(
                    tf.truncated_normal([output_size, 1],
                                        stddev=1.0 / math.sqrt(float(output_size)),
                    dtype=tf.float64),
                    name='weights')
                biases = tf.Variable(tf.zeros([1],dtype=tf.float64),
                                     name='biases')
                    
                pos_stage_1 = tf.matmul(tf_output, weights) + biases
    
                if (key_size == 1):
                    pos_stage_1 = tf.reshape(pos_stage_1,[-1])


                # At this point we want the model to have produced
                # output in the range [-0.5, 0.5], but we want the
                # final output to be in the range [0,N), so we need
                # to add 0.5 and multiply by N.
                # Doing normalization this way can effect how
                # the learning rates scale with N, so we should
                # consider doing this normalization outside of
                # the Tensflow pipeline.
                pos_stage_1 = tf.scalar_mul(tf.constant(self._data_set.num_positions,
                                                        dtype=tf.float64),
                                            tf.add(pos_stage_1,
                                                   tf.constant(0.5,dtype=tf.float64)))
                    
                pos_stage_1 = tf.identity(pos_stage_1,name="pos")
    
        return pos_stage_1


    def _setup_loss_stage_1(self, pos_stage_1, pos_true):        
        """Calculates the loss from the keys and positions, for Stage 1.
        Args:
        pos_stage_1: int64 tensor with shape [batch_size, 1].
                     The position predicted in stage 1
        pos_true: int64 tensor wiht shape [batch_size].
                  The true position for the key.
        Returns:
        loss: Loss tensor, using mean_squared_error.
        """
        labels = tf.to_int64(pos_true)
        loss = tf.losses.mean_squared_error(
            labels=pos_true,
            predictions=pos_stage_1)
        
        return loss


    def _setup_training_stage_1(self, loss):
        """Sets up the TensorFlow training operations for Stage 1.

        Args:
            loss: loss tensor, from self._setup_loss_stage_1()

        Returns:
            train_op: the TensorFlow operation for training Stage 1.
        """

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)

        # Create optimizer with the given learning rate.
        # AdamOptimizer is used, but other other optimizers could
        # have been chosen (e.g. the commented-out examples).
        optimizer = tf.train.AdamOptimizer(self.learning_rates[0])
        #optimizer = tf.train.AdadeltaOptimizer(self.learning_rates[0])
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rates[0])
        
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
        
        return train_op


    def _setup_inference_stage_2(self, keys, pos_stage_1):
        """Set up Stage 2 inference.

        Args:
            keys: TensorFlow placeholder for keys
            pos_stage_1: tensor, output of Stage 1 inference
        Returns:
            pos_stage_2: tensor, output of Stage 2 inference
        """

        max_index = self._data_set.num_positions

        # Stage 2 
        with tf.name_scope('stage_2'):

            keys_std = self._data_set.keys_std
            keys_mean = self._data_set.keys_mean

            keys = tf.squeeze(keys,1)
            keys = tf.identity(keys,name='key')
            keys = tf.cast(keys,dtype=tf.float64)
                
            # Normalize using mean and standard deviation
            keys_normed = tf.scalar_mul(tf.constant(1.0/keys_std),
                                        tf.subtract(keys,tf.constant(keys_mean)))
            
            # Normalize further by dividing by 2*sqrt(3), so that
            # a uniform distribution in the range [a,b] would transform
            # to a uniform distribution in the range [-0.5,0.5]
            
            keys_normed = tf.scalar_mul(tf.constant(0.5/np.sqrt(3)),
                                        keys_normed)

            # Calculate which expert to use
            expert_index = tf.to_int32(
                tf.floor(
                    tf.scalar_mul(tf.constant(self._expert_factor,dtype=tf.float64),
                                  pos_stage_1)))
    
            # Ensure that expert_index is within range [0,self.num_experts)
            expert_index = tf.maximum(tf.constant(0),expert_index)
            expert_index = tf.minimum(tf.constant(self.num_experts-1),expert_index)
            expert_index = tf.identity(expert_index, name="expert_index")
    
                
            # Explicitly handle batches
            num_batches  = tf.shape(pos_stage_1)[0]
            num_batches = tf.identity(num_batches, name="num_batches")
            expert_index_flat = (tf.reshape(expert_index, [-1])
                                 + tf.range(num_batches) * self.num_experts)
            expert_index_flat = tf.identity(expert_index_flat, name="expert_index_flat")

            # This version uses tf.unsroted_segment_sum
            gates_flat = tf.unsorted_segment_sum(
                tf.ones_like(expert_index_flat), 
                expert_index_flat, 
                num_batches * self.num_experts)
            gates = tf.reshape(gates_flat, [num_batches, self.num_experts],
                               name="gates")

            # This version uses SparseTensor, and could potential replace the
            # previous block of code, but it doesn't work yet
            # 
            #expert_index_flat = tf.reshape(expert_index_flat,[-1,1])
            #gates_flat = tf.SparseTensor(tf.cast(expert_index_flat,dtype=tf.int64),
            #                             tf.ones([self.num_experts]),
            #                             dense_shape=[self.num_experts*num_batches,1])
            #gates = tf.sparse_reshape(gates_flat, [num_batches, tf.constant(self.num_experts)],
            #                          name="gates")
            #gates = tf.sparse_tensor_to_dense(gates)

            # Name the gates for later access
            gates = tf.cast(gates,dtype=tf.float64)
            gates = tf.identity(gates, name="gates")

         
            # Normalize variable weights and biases
            weights = tf.Variable(
                tf.truncated_normal([self.num_experts],
                                    mean=1.0*max_index,
                                    stddev=0.5*max_index,
                                    dtype=tf.float64),
                name='weights')

            biases = tf.Variable(tf.zeros([self.num_experts],dtype=tf.float64),
                                 name='biases')

            # Dot-product gates with weights and biases,
            # to only use one expert at a time.
            gated_weights = tf.multiply(gates,weights)
            gated_biases = tf.multiply(gates,biases)
            gated_weights_summed = tf.reduce_sum(gated_weights,axis=1)
            gated_biases_summed = tf.reduce_sum(gated_biases,axis=1)

            # Name the variables for later access
            gated_weights = tf.identity(gated_weights, name="gated_weights")
            gated_biases = tf.identity(gated_biases, name="gated_biases")
            gated_weights_summed = tf.identity(gated_weights_summed, name="gated_weights_summed")
            gated_biases_summed = tf.identity(gated_biases_summed, name="gated_biases_summed")

            # Do the linear regression to predict the key position
            pos_stage_2 = tf.add( tf.multiply(keys_normed, gated_weights_summed), gated_biases_summed)
            pos_stage_2 = tf.identity(pos_stage_2, name="pos")

        # Returns the predicted position for Stage 2
        return pos_stage_2

    
    def _setup_loss_stage_2(self, pos_stage_2, pos_true):
        """Calculates the loss from the keys and positions, for Stage 2.
        
        Args:
            pos_stage_2: int64 tensor with shape [batch_size, 1].
                         The position predicted in stage 1
            pos_true: int64 tensor wiht shape [batch_size].
                      The true position for the key.
        Returns:
            loss: Loss tensor, using mean_squared_error.
        """
        # Stage 2 
        with tf.name_scope('stage_2'):
            labels = tf.to_int64(pos_true)
            loss = tf.losses.mean_squared_error(
                labels=pos_true,
                predictions=pos_stage_2)
            
        return loss


    def _setup_training_stage_2(self, loss):
        """Sets up the TensorFlow training operations for Stage 2.

        Args:
            loss: loss tensor, from self._setup_loss_stage_2()

        Returns:
            train_op: the TensorFlow operation for training Stage 2.
        """

        # Stage 2 
        with tf.name_scope('stage_2'):
            # Add a scalar summary for the snapshot loss.
            tf.summary.scalar('loss', loss)

            
            # Create optimizer with the given learning rate.
            # Uses AdamOptimizer, but others could be considered
            # (e.g. see commented-out examples)
            optimizer = tf.train.AdamOptimizer(self.learning_rates[1])
            #optimizer = tf.train.AdadeltaOptimizer(self.learning_rates[1])
            #optimizer = tf.train.GradientDescentOptimizer(self.learning_rates[1])

            # Create a variable to track the global step.
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # Get list of variables needed to train stage 2
            variables_stage_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stage_2')
            # Use the optimizer to apply the gradients that minimize the loss
            # (and also increment the global step counter) as a single training step.
            train_op = optimizer.minimize(loss, global_step=global_step, var_list=variables_stage_2)

            return train_op

    
    def run_training(self,
                     batch_sizes=None,
                     max_steps=None,
                     learning_rates=None,
                     model_save_dir=None):
        """Train both Stage 1 and Stage 2 (in order)

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
            No output, but prints training information to stdout.
        """


        # First update model with new batch_sizes, learning_rates, max_steps,
        # and model_save_dir
        if batch_sizes is not None:
            self.batch_sizes = batch_sizes
        if learning_rates is not None:
            self.learning_rates = learning_rates
        if max_steps is not None:
            self.max_steps = max_steps
        if model_save_dir is not None:
            self.model_save_dir = model_save_dir
        
        # Reset the default graph  
        tf.reset_default_graph()
    
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():

            ## Stage 1
            
            # Generate placeholders for the images and labels.
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(self.batch_sizes[0])
    
            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            # Add to the Graph the Ops for loss calculation.
            loss_s1 = self._setup_loss_stage_1(pos_stage_1, labels_placeholder)
            
            # Add to the Graph the Ops that calculate and apply gradients.
            train_op_s1 = self._setup_training_stage_1(loss_s1)

            # Currently no need for Summaries, but could add this later
            # Build the summary Tensor based on the TF collection of Summaries.
            #summary = tf.summary.merge_all()
            
                        
            ## Stage 2

            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1) 
            
            # Add to the Graph the Ops for loss calculation.
            loss_s2 = self._setup_loss_stage_2(pos_stage_2, labels_placeholder)
            
            # Add to the Graph the Ops that calculate and apply gradients.
            train_op_s2 = self._setup_training_stage_2(loss_s2)
            
            
            ## Done with Stage definitions
            
            # Add the variable initializer Op.
            init = tf.global_variables_initializer()
            
            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()
            
            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Could use a SummaryWrite in future implementation
            # Instantiate a SummaryWriter to output summaries and the Graph.
            #summary_writer = tf.summary.FileWriter(model_save_dir, sess.graph)
            
            # And then after everything is built:
            
            # Run the Op to initialize the variables.
            sess.run(init)
            

            ## Train Stage 1 
            print("Stage 1 Training:")
            
            training_start_time = time.time()
            
            # Start the training loop.
            for step in xrange(self.max_steps[0]):
                start_time = time.time()
                
                # Fill a feed dictionary with the actual set of keys and labels
                # for this particular training step.
                feed_dict = self._fill_feed_dict(keys_placeholder,
                                                 labels_placeholder,
                                                 batch_size=self.batch_sizes[0])
                
                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.
                _, loss_value = sess.run([train_op_s1, loss_s1],
                                         feed_dict=feed_dict)
                
                duration = time.time() - start_time

                # Print an overview fairly often.
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec, total %.3f secs)' % (step, np.sqrt(loss_value), duration, time.time() - training_start_time))
                    # Could write summary info in future implementation.
                    # Update the events file.
                    #summary_str = sess.run(summary, feed_dict=feed_dict)
                    #summary_writer.add_summary(summary_str, step)
                    #summary_writer.flush()
                    
                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 10000 == 0 and (step + 1) != self.max_steps[0]:
                    checkpoint_file = os.path.join(self.model_save_dir, 'stage_1.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                if (step + 1) == self.max_steps[0]:
                    checkpoint_file = os.path.join(self.model_save_dir, 'stage_1.ckpt')
                    saver.save(sess, checkpoint_file)

            ## Train Stage 2
            print("\nStage 2 Training:")
            
            # Start the training loop.
            for step in xrange(self.max_steps[1]):
                start_time = time.time()
                
                # Fill a feed dictionary with the actual set of keys and labels
                # for this particular training step.
                feed_dict = self._fill_feed_dict(keys_placeholder,
                                                 labels_placeholder,
                                                 batch_size=self.batch_sizes[1])
                
                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.
                _, loss_value = sess.run([train_op_s2, loss_s2],
                                         feed_dict=feed_dict)
                
                duration = time.time() - start_time
                
                # Print an overview fairly often.
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec, total %.3f secs)' % (step, np.sqrt(loss_value), duration, time.time() - training_start_time))
                    # Could write summary info in future implementation.
                    # Update the events file.
                    #summary_str = sess.run(summary, feed_dict=feed_dict)
                    #summary_writer.add_summary(summary_str, step)
                    #summary_writer.flush()
                    
                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 10000 == 0 and (step + 1) != self.max_steps[1]:
                    checkpoint_file = os.path.join(self.model_save_dir, 'stage_2.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                if (step + 1) == self.max_steps[1]:
                    checkpoint_file = os.path.join(self.model_save_dir, 'stage_2.ckpt')
                    saver.save(sess, checkpoint_file)



    def _run_inference_tensorflow(self,keys):
        """Run inference using TensorFlow checkpoint

        Args:
            keys: numpy array of one or more keys

        Returns:
            pos_stage_2: numpy array of predicted position for each key.
            expert: numpy array of expert used for each key.
        """

        
        batch_size = keys.shape[0]
        
        # Reset the default graph  
        tf.reset_default_graph()
    
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():

            ## Stage 1
            
            # Generate placeholders for the keys and labels.
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(batch_size)
    
            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            ## Stage 2
            
            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1) 
            
            ## Done with Stage definitions
            
            # Add the variable initializer Op.
            init = tf.global_variables_initializer()
            
            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()
            
            # Create a session for running Ops on the Graph.
            sess = tf.Session()
            
            # Run the Op to initialize the variables.
            sess.run(init)

            # Load trained variables
            checkpoint_file = os.path.join(self.model_save_dir, "stage_2.ckpt")
            meta_file = os.path.join(self.model_save_dir,"stage_2.ckpt.meta")
            saver = tf.train.import_meta_graph(meta_file)    
            saver.restore(sess, checkpoint_file)

            # Fill a feed dictionary with keys
            feed_dict = {keys_placeholder: keys}
               
            # Get the expert for each key
            expert_index = sess.graph.get_tensor_by_name("stage_2/expert_index:0")
            experts = sess.run(expert_index,feed_dict=feed_dict)

            # Get the predicted position for each key
            stage_2_out = sess.graph.get_tensor_by_name("stage_2/pos:0")
            pos = sess.run(stage_2_out,feed_dict=feed_dict)
        
        return (pos, experts)


    def inspect_inference_steps(self, keys):
        """Run inference using TensorFlow, and print out important tensor
           values. Can be useful for debugging.

        Args:
            keys: numpy array of one or more keys

        Returns:
           Prints the values of several model tensors to stdout.
        """

        
        batch_size = keys.shape[0]
        
        # Reset the default graph  
        tf.reset_default_graph()
    
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():

            ## Stage 1
            
            # Generate placeholders for the keys and labels.
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(batch_size)
    
            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            ## Stage 2
            
            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1) 
            
            ## Done with Stage definitions
            
            # Add the variable initializer Op.
            init = tf.global_variables_initializer()
            
            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()
            
            # Create a session for running Ops on the Graph.
            sess = tf.Session()
            
            # Run the Op to initialize the variables.
            sess.run(init)

            # Load trained variables
            checkpoint_file = os.path.join(self.model_save_dir, "stage_2.ckpt")
            meta_file = os.path.join(self.model_save_dir,"stage_2.ckpt.meta")
            saver = tf.train.import_meta_graph(meta_file)    
            saver.restore(sess, checkpoint_file)

            # Fill a feed dictionary with keys
            feed_dict = {keys_placeholder: keys}

            # Print the values of tensors used in the model
            
            print("Stage 1 position predictions (one per batch):")
            print(sess.run(pos_stage_1,feed_dict=feed_dict))
            
            print("Expert Index (one per batch):")
            expert_index = sess.graph.get_tensor_by_name("stage_2/expert_index:0")
            print(sess.run(expert_index,feed_dict=feed_dict))
            
            print("Expert Index Flat (all batches):")
            expert_index_flat = sess.graph.get_tensor_by_name("stage_2/expert_index_flat:0")
            print(sess.run(expert_index_flat,feed_dict=feed_dict))

            print("Gate vector (one per batch):")
            gates = sess.graph.get_tensor_by_name("stage_2/gates:0")
            print(sess.run(gates,feed_dict=feed_dict))
            
            print("Gate vector times weights (one per batch):")
            gated_weights = sess.graph.get_tensor_by_name("stage_2/gated_weights:0")
            print(sess.run(gated_weights,feed_dict=feed_dict))
            
            print("Gate vector times weights summed (one per batch):")
            gated_weights_summed = sess.graph.get_tensor_by_name("stage_2/gated_weights_summed:0")
            print(sess.run(gated_weights_summed,feed_dict=feed_dict))
            
            print("Gate vector times biases (one per batch):")
            gated_biases = sess.graph.get_tensor_by_name("stage_2/gated_biases:0")
            print(sess.run(gated_biases,feed_dict=feed_dict))
            
            print("Gate vector times biases summed (one per batch):")
            gated_biases_summed = sess.graph.get_tensor_by_name("stage_2/gated_biases_summed:0")
            print(sess.run(gated_biases_summed,feed_dict=feed_dict))
            
            print("Key (one per batch):")
            key = sess.graph.get_tensor_by_name("stage_2/key:0")
            print(sess.run(key,feed_dict=feed_dict))
        
            print("Stage 2 position prediction = w*key + b (one per batch):")
            stage_2_out = sess.graph.get_tensor_by_name("stage_2/pos:0")
            print(sess.run(stage_2_out,feed_dict=feed_dict))

    
    def get_weights_from_trained_model(self):
        """Retrieves weights and biases from TensorFlow checkpoints. 
           Stores the weights and biases in class member variables, to be used
           for faster inference calculations (such as using numpy).

        Args:
            -

        Returns:
            -
        """
        
        # Reset the default graph  
        tf.reset_default_graph()
    
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():

             # Generate placeholders for the keys and labels.
            batch_size = 1
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(1)
            ## Stage 1
            
            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            ## Stage 2
            
            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1) 
            
            # Add the variable initializer Op.
            init = tf.global_variables_initializer()
            
            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()
            
            # Create a session for running Ops on the Graph.
            sess = tf.Session()
            
            # Run the Op to initialize the variables.
            sess.run(init)

            checkpoint_file = os.path.join(self.model_save_dir, "stage_2.ckpt")
            meta_file = os.path.join(self.model_save_dir,"stage_2.ckpt.meta")
            saver = tf.train.import_meta_graph(meta_file)    
            saver.restore(sess, checkpoint_file)


            # Get the weights and biases variables, and store them
            
            for layer_idx in range(0,len(self.hidden_layer_widths)):

                name_scope = "stage_1/hidden_" + str(layer_idx+1) 

                weights = sess.graph.get_tensor_by_name(name_scope + "/weights:0") 
                self.hidden_w[layer_idx] = sess.run(weights)

                biases = sess.graph.get_tensor_by_name(name_scope + "/biases:0") 
                self.hidden_b[layer_idx] = sess.run(biases)

            linear_w = sess.graph.get_tensor_by_name("stage_1/linear/weights:0")
            self.linear_w = sess.run(linear_w)

            linear_b = sess.graph.get_tensor_by_name("stage_1/linear/biases:0")
            self.linear_b = sess.run(linear_b)

            stage_2_w = sess.graph.get_tensor_by_name("stage_2/weights:0")
            self.stage_2_w = sess.run(stage_2_w)

            stage_2_b = sess.graph.get_tensor_by_name("stage_2/biases:0")
            self.stage_2_b = sess.run(stage_2_b)

            
    def time_inference_tensorflow(self,N=100):
        """Calculates time per inference using TensorFlow, not counting the time
           it takes to start a sessions and load the graph.

        Args:
            N: Number of time to run inference to get an average.

        Returns:
            Time (in seconds) to run inference on one batch.
        """

        # Only test batch_size of 1.
        # Future implementations should consider timing larger batches.
        
        batch_size = 1
        
        # Reset the default graph  
        tf.reset_default_graph()
    
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():

            ## Stage 1
            
            # Generate placeholders for the images and labels.
            keys_placeholder, labels_placeholder = self._setup_placeholder_inputs(batch_size)    
            # Build a Graph that computes predictions from the inference model.
            pos_stage_1 = self._setup_inference_stage_1(keys_placeholder)

            ## Stage 2
            
            pos_stage_2 = self._setup_inference_stage_2(keys_placeholder, pos_stage_1) 
            
            ## Done with Stage definitions
            
            # Add the variable initializer Op.
            init = tf.global_variables_initializer()
            
            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()
            
            # Create a session for running Ops on the Graph.
            sess = tf.Session()
            
            # Run the Op to initialize the variables.
            sess.run(init)

            # Load checkpoint
            checkpoint_file = os.path.join(self.model_save_dir, "stage_2.ckpt")
            meta_file = os.path.join(self.model_save_dir,"stage_2.ckpt.meta")
            saver = tf.train.import_meta_graph(meta_file)    
            saver.restore(sess, checkpoint_file)


            # Time N inference steps
            
            start_time = time.time()
            for n in range(N):

                key = self._data_set.keys[n]
                
                # Fill a feed dictionary with the set of keys

                feed_dict = {keys_placeholder: [key]}
                
                expert_index = sess.graph.get_tensor_by_name("stage_2/expert_index:0")
                experts = sess.run(expert_index,feed_dict=feed_dict)
                
                stage_2_out = sess.graph.get_tensor_by_name("stage_2/pos:0")
                pos = sess.run(stage_2_out,feed_dict=feed_dict)

                
        return (time.time() - start_time)/N

        
    
    def calc_min_max_errors(self,
                            key_pos=None,
                            batch_size=10000):
        """Calculates the errors each Stage 2 expert makes in predicting the 
           keys poistion. Inference is run on the full data set to get the errors.
           The calculated prediction errors are stored in class member variables.

        Args:
            key_pos: Numpy array of (key,position) pairs for which to calculate errors. 
                     If key_pos==None, then all keys are used.
            batch_size: integer size of batches.

        Returns:
            -
        """

        if key_pos == None: # Use all keys

            # Initialize errors
            self._initialize_errors()

            # Use all keys and positions
            keys = self._data_set.keys
            true_positions = self._data_set.positions
            num_keys = self._data_set.num_keys

        else: # Use subset of keys

            # Only use keys and position specified by key_pos
            keys, true_positions = list(zip(*key_pos))
            keys = list(keys)
            true_positions = list(true_positions)
            num_keys = len(keys)
        
            
        # Calculate errors for each expert
        for step in range(0, num_keys, batch_size):
        
            positions, experts = self.run_inference(keys[step:(step+batch_size)])
            true_positions_batch = true_positions[step:(step+batch_size)]
            
            for idx in range(len(positions)):
                
                pos = np.round(positions[idx])
                expert = experts[idx]
                true_pos = true_positions_batch[idx]
                
                
                self.min_predict[expert] = np.minimum(self.min_predict[expert],
                                                      pos)
                self.max_predict[expert] = np.maximum(self.max_predict[expert],
                                                      pos)
                
                self.min_pos[expert] = np.minimum(self.min_pos[expert],
                                                  true_pos)
                self.max_pos[expert] = np.maximum(self.max_pos[expert],
                                                  true_pos)
                
                error = pos - true_pos
                if error > 0:
                    self.max_error_left[expert] = np.maximum(self.max_error_left[expert],
                                                             error)
                elif error < 0:
                    self.max_error_right[expert] = np.maximum(self.max_error_right[expert],
                                                              np.abs(error))

                
                
                        
    def _initialize_errors(self):
        """Helper function that initializes all errors before call to 

        Args:
            --

        Returns:
            --
        
        """
        
        # Initialize errors for each expert
            
        # The maximum left and right error for each expert
        self.max_error_left = np.zeros([self.num_experts])
        self.max_error_right = np.zeros([self.num_experts])
        
        # The minimum and maximum position predictions of each expert
        self.min_predict = (np.ones([self.num_experts]) * self._data_set.num_positions) - 1
        self.max_predict = np.zeros([self.num_experts])
        
        # The minimum and maximum true positions handled by each expert
        self.min_pos = (np.ones([self.num_experts]) * self._data_set.num_positions) - 1
        self.max_pos = np.zeros([self.num_experts])
        
        
    def _run_inference_numpy_0_hidden(self,keys):
        """Run inference using numpy, assuming 0 hidden layers in Stage 1.

        Args:
            keys: List or numpy array of keys.

        Returns:
            (pos_stage_2, experts)

            pos_stage_2: Position predictions for the keys.

            experts: Experts used for the keys.
        
        """

        # Do the same calculations found in self._setup_inference_stage1()
        # and in self._setup_inference_stage1(), but use numpy instead of
        # TensorFlow.
        
        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor
                
        out = np.matmul(keys,self.linear_w)
        out = np.add(out,self.linear_b)

        out = np.add(out,0.5)
        out = np.multiply(out,self._data_set.num_positions)
        
        expert = np.multiply(out,self._expert_factor) 
        expert = expert.astype(np.int32) # astype() equivalent to floor() + casting
        expert = np.maximum(0,expert)
        expert = np.minimum(self.num_experts-1,expert)

        out = np.multiply(keys,self.stage_2_w[expert])
        out = np.add(out,self.stage_2_b[expert])
        
        return (out, expert)

    def _run_inference_numpy_0_hidden_0_experts(self,keys):
        """Run inference using numpy, assuming 0 hidden layers in Stage 1.

        Args:
            keys: numpy array of keys.

        Returns:

            pos_stage_1: Position predictions for the keys.

        """

        # Do the same calculations found in self._setup_inference_stage_1()
        # and in self._setup_inference_stage_2(), but use numpy instead of
        # TensorFlow.
        
        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor
                
        out = np.matmul(keys,self.linear_w)
        out = np.add(out,self.linear_b)

        out = np.add(out,0.5)
        out = np.multiply(out,self._data_set.num_positions)
        
        return out

    
    def _run_inference_numpy_1_hidden(self,keys):
        """Run inference using numpy, assuming 1 hidden layers in Stage 1.

        Args:
            keys: List or numpy array of keys.

        Returns:
            (pos_stage_2, experts)

            pos_stage_2: Position predictions for the keys.

            experts: Experts used for the keys.
        
        """

        # Do the same calculations found in self._setup_inference_stage1()
        # and in self._setup_inference_stage1(), but use numpy instead of
        # TensorFlow.

        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor
                
        out = np.matmul(keys,self.hidden_w[0])
        out = np.add(out,self.hidden_b[0])
        out = np.maximum(0.0,out)

        out = np.matmul(out,self.linear_w)
        out = np.add(out,self.linear_b)

        out = np.add(out,0.5)
        out = np.multiply(out,self._data_set.num_positions)

        expert = np.multiply(out,self._expert_factor) 
        expert = expert.astype(np.int32) # astype() equivalent to floor() + casting
        expert = np.maximum(0,expert)
        expert = np.minimum(self.num_experts-1,expert)

        out = np.multiply(keys,self.stage_2_w[expert])
        out = np.add(out,self.stage_2_b[expert])
        
        return (out, expert)


    def _run_inference_numpy_2_hidden(self,keys):
        """Run inference using numpy, assuming 2 hidden layers in Stage 1.

        Args:
            keys: List or numpy array of keys.

        Returns:
            (pos_stage_2, experts)

            pos_stage_2: Position predictions for the keys.

            experts: Experts used for the keys.
        
        """

        # Do the same calculations found in self._setup_inference_stage1()
        # and in self._setup_inference_stage1(), but use numpy instead of
        # TensorFlow.

        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor
                
        out = np.matmul(keys,self.hidden_w[0])
        out = np.add(out,self.hidden_b[0])
        out = np.maximum(0.0,out)

        out = np.matmul(out,self.hidden_w[1])
        out = np.add(out,self.hidden_b[1])
        out = np.maximum(0.0,out)

        out = np.matmul(out,self.linear_w)
        out = np.add(out,self.linear_b)

        out = np.add(out,0.5)
        out = np.multiply(out,self._data_set.num_positions)

        expert = np.multiply(out,self._expert_factor) 
        expert = expert.astype(np.int32) # astype() equivalent to floor() + casting
        expert = np.maximum(0,expert)
        expert = np.minimum(self.num_experts-1,expert)

        out = np.multiply(keys,self.stage_2_w[expert])
        out = np.add(out,self.stage_2_b[expert])
        
        return (out, expert)

    
    def _run_inference_numpy_n_hidden(self,keys):
        """Run inference using numpy, assuming any number of hidden layers in Stage 1.

        Args:
            keys: List or numpy array of keys.

        Returns:
            (pos_stage_2, experts)

            pos_stage_2: Position predictions for the keys.

            experts: Experts used for the keys.
        
        """

        # Do the same calculations found in self._setup_inference_stage1()
        # and in self._setup_inference_stage1(), but use numpy instead of
        # TensorFlow.

        keys = (keys - self._keys_mean) * self._keys_std_inverse
        keys = keys * self._keys_norm_factor

        out = keys
        for layer_idx in range(0,len(self.hidden_layer_widths)):
                
                out = np.matmul(out,self.hidden_w[layer_idx])
                out = np.add(out,self.hidden_b[layer_idx])
                out = np.maximum(0.0,out)

        out = np.matmul(out,self.linear_w)
        out = np.add(out,self.linear_b)

        out = np.add(out,0.5)
        out = np.multiply(out,self._data_set.num_positions)

        expert = np.multiply(out,self._expert_factor) 
        expert = expert.astype(np.int32) # astype() equivalent to floor() + casting
        expert = np.maximum(0,expert)
        expert = np.minimum(self.num_experts-1,expert)

        out = np.multiply(keys,self.stage_2_w[expert])
        out = np.add(out,self.stage_2_b[expert])
        
        return (out, expert)

