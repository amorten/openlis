# openLIS
A Python package that implements a Recursive-Model Index (RMI) and provides a high-level database-like interface.

## Description

This package implements the Recursive-Model Index described in the paper 
"The Case for Learned Index Structures," 
[Kraska et al., 2017](http://arxiv.org/abs/1712.01208)
([pdf](http://arxiv.org/pdf/1712.01208.pdf)). 

In addition to implementing the Recursive-Model Index, openLIS provides
a high-level database-like interface with Select, Insert, and Delete functionality.

### The Recursive-Model Index

An example of a standard index structure is a BTree, which keeps `keys` sorted within a tree structure and allows fast lookup of the `keys`. BTrees and their variants are often used by modern databases to perform fast search and are particularly efficient when searching for a range of `keys`.

The basic idea of the Learned Index Structures paper is to replace standard index structures (such as BTrees) with statistical learning methods (such as neural networks). Both take a `key` as input and output the position of the key in the data set.
        
The paper proposes using a Recursive-Model Index (RMI) as the statistical learning method. A general RMI would consist of several stages. We only implement two stages, as was done in the paper.

In Stage 1 a neural network is used to predict the position of the `key`. The predicted position is then used to choose which linear regression model to use in Stage 2. Each linear regression model in Stage 2 is called an "expert." Each expert handles only a small subset of the outputs of Stage 1. For more detail see the paper referenced above.

The output of Stage 2 is a (hopefully) improved prediction of the position of the `key`. Because the predicted position is typically slightly off, a binary search over nearby locations is used to find the `key`. The range of the binary search is determined by the known maximum error of each linear regression model.





## Getting Started

You can install this package using pip:  

```
pip install openlis
```

## Example usage

For a basic example, see the Jupyter notebook:  

[openlis_example.ipynb](openlis_example.ipynb)

The notebook demonstrates generating a random data set, training the Recursive-Model Index on that data set, and using the three database functions: Select, Insert and Delete.

## Usage details

### Generating and loading data

The module defines a class DataSet. You can obtain a DataSet object in four ways:

1) You can generate a random dataset using

```
data_set = openlis.data.generate_uniform_floats(num_keys,
                                          key_range,
                                     iseed=17)
```

or

```
openlis.data.generate_normal_floats(num_keys,
                                    mean=0.0,
                                    std=1.0,
                                    iseed=17)
```

The former uses a uniform distribution, whereas the latter uses a normal distribution.

2)  Or, first create a Numpy array of keys. Save those keys using `numpy.save()`. Then create a DataSet object using

```
data_set = openlis.data.load_keys_numpy(dir,fname)
```

3) Create your own DataSet from a list of keys using

```
data_set = DataSet(keys)
```

4) Finally, you may create your own DataSet from a list of keys and a list of labels using

```
data_set = DataSet(keys, labels)
```
This final method should be used with caution, because you are required to ensure that `labels` are the sorted positions of the `keys`. It is better to use option 3 above, which automatically generates the `labels` for you.

#### Notes:
* The keys are assumed to be floating point values. If you use something else (like integers), some functions might not work. A future implementation should be able to handle different data types.
* Also, a real data set would have `keys` paired with actual data (or pointers to the data). Here we just deal with `keys`, keeping in mind that we could later implement a data array in parallel with `keys`.


### Creating the training DataSet

From the `data_set` created above, we must split the data into training and validation. For our problem, because we are not particularly concerned with overfitting, it typically makes sense to use 100% of the data for training, so we set `validation_size=0` (the Default value).

Use the following function to create the training and validation data sets:

```
data_sets = li.data.create_train_validate_data_sets(data_set,
                                        validation_size=0)
```

The  training DataSet will be stored in 
```
data_sets.train
```
and the validation DataSet will be stored in
```
data_sets.valid
```

### Create a Recursive-Index-Model

Once we have the training data set, we can create the Recursive-Index-Model based on the training data:

```
rmi = openlis.model.RMI_simple(data_sets.train,
                               hidden_layer_widths=[8,8], 
                               num_experts=100)
```

In the above example, the Stage 1 neural network will have two hidden layers each with width=8, and Stage 2 will have 100 experts. 

The number and widths of the hidden layers are set by the input parameter `hidden_layer_widths`. For example, to create three layers with widths 100, 50, and 10, set

```
hidden_layer_widths = [100,50,10]
```

If you want zero hidden layers (i.e. just do a linear regression with no activation), set

```
hidden_layer_widths = []
```

The number of linear regression models used in Stage 2 is set by the input parameter `num_experts`. For example, if you want 100 experts, set

```
num_experts = 100
```
#### Notes:
* The number of experts in Stage 2 must be at least 1. 
* Currently not implemented is the possibility of having no Stage 2. 

### Create the database-like interface

Once a model is defined, we can create a database-like interface to that model using either 

```
rmi_db = openlis.database.IndexStructurePacked(model=rmi)

```

or

```
rmi_db = openlis.database.IndexStructureGapped(model=rmi, scale=2)

```

The former creates an object of type `IndexStructurePacked` while the latter creates an object of type `IndexStructureGapped`. 

The word "Packed" in `IndexStructurePacked` refers to how the keys are stored: the keys are simply stored contiguously in a Numpy array. This means that insertions and deletions will be quite slow, because the array must be resized. 

The word "Gapped" in `IndexStructureGapped` also refers to how the keys are stored: gaps are left between keys in the Numpy array. The gaps leave room for fast key insertions and require no additional work for key deletions. The parameter `scale` sets the size of the "gapped" array relative to a packed array. For example, setting `scale = 3` would create two gaps for every key, evenly distributed between the keys.

If using `IndexStructureGapped`, the fuction

```
rmi_db.rescale(scale=3)
```
can be used at any time to rescale the array. 

### Train the model

Training the model (to predict the positions of the keys) can be accomplished via the database interface:

```
rmi_db.train(batch_sizes=[10000,1000],
             max_steps=[500,500],
             learning_rates=[0.001,1000],
             model_save_dir='tf_checkpoints_example')
``` 

`batch_sizes` are the batch sizes for the two stages.  
`max_steps` are the maximum number of batches for each stage.  
`learning_rates` are the learning rates for the two stages.  
`model_save_dir` is where to save the trained model.

Currently, the user must choose the above hyperparameters manually. 

Training proceeds as follows:

* The Stage 1 model is trained with Adam optimization, using Mean Squared Error (MSE) as Loss.
*  The Stage 2 model is trained with Stage 1 variables held fixed. Stage 2 also uses MSE and an Adam optimizer.

Then `rmi_db.train(...)` also performs the following steps:

*  The maximum errors for each linear regression  are calculated and stored.
*  The trained variables are retrieved from the TensorFlow model and stored as Numpy arrays (for use in faster inference calculations).

If you prefer not to use the database interface, you can train the model directly with

```
rmi.run_training(batch_sizes=[10000,1000],
                 max_steps=[500,500],
                 learning_rates=[0.001,1000],
                 model_save_dir='tf_checkpoints_example')
```
Then you can calculate the linear regression errors with

```
rmi.calc_min_max_errors(batch_size=10000)
```
And then you may retrieve the trained variables from the TensorFlow model:

```
rmi.get_weights_from_trained_model()
```

Basically, `rmi_db.train(...)` wraps the above three functions into a single convenient function call.

### Use the Database

Once we have a fully trained model with saved errors and saved weights -- for example, after we have run `rmi_db.train(...)` -- we are ready to use the database functionality!

Given a list or Numpy array `keys`, you may  


* Select the `keys`:

	``` 
	pos = rmi_db.select(keys)
	```

	The output, `pos`, is a Numpy array containing the
	positions of the keys. By convention, `pos[i] = -1`
	if `keys[i]` is not in the data set.

* Insert the `keys`:

	```
	success = rmi_db.insert(keys)
	```
	The output, `success`, is a boolean array that
	indicates whether each key insertion was a success. 
	Insertion fails if the key is already in the data 
	set (i.e. uniqueness of keys is enforced).
	
* Delete the `keys`:

	```
	success = rmi_db.delete(keys)
	```
	The output, `success`, is a boolean array that 
	indicates whether each key deletion was a success. 
	Deletion fails if the key was not in the data set.
	
#### Notes:
* Instead of a list or Numpy array of `keys`, you may input a single `key` to any of the above functions.
* When using `IndexStructurePacked`, submitting multiple keys at a time is significantly faster than submitting single keys repeatedly.
     * Select, Insert and Delete all run inference on all of the input `keys` as a single batch, which can be faster than running inference on one key at a time.
     * Insert and Delete, as currently implemented in `IndexStructurePacked` require O(n+k) time, where n is the size of the data set and k is the number of keys to insert or delete. Repeated single-key insertions or deletions would instead require O(kn) time.
* When using `IndexStructureGapped`, key insertion fails when the array runs out of gaps (and a warning is sent to stdout). The user should periodically use `rmi_db.rescale(...)` to create additional gaps in the key array. Having a sufficient number of gaps is important for fast insertion.

####Notes:
* In a future implementation of `IndexStructurePacked`, perhaps the key array could expand automatically whenever the gaps become too sparse after a call to `rmi_db.insert(...)`.

### Don't forget to retrain the model!

After a significant number of insertions or deletions, the positions of the keys may differ substantially from their initial positions. Thus, you should periodically retrain the model with

```
rmi_db.train()
```

The hyperparameters from the previous training are remembered and used by default.

#### Notes:
* In a future implementation, the user could receive a warning when the error of the model exceeds some threshold.

### Save and load the database

Saving and loading a database is easy. To save a database use

```
openlis.database.save_db(rmi_db,"temp_db.p")
```

To load a saved database use

```
rmi_db = openlis.database.load_db("temp_db.p")
```

The filename `"temp_db.p"` used above is just an example -- you can use any filename you want. Because saving and loading use the `pickle` package, using a filename extension `.p` is a reasonable choice.

### Other functions

Some functions do not fit within the standard use case.  For example, model inference can be run directly using

```
rmi.run_inference(keys)
```
rather than indirectly via the Select, Insert or Delete functions.  

The function `run_inference()` actually aliases one of several functions that use Numpy to run optimized inference. For example, when 1 hidden layer is used, then `rmi.run_inference()` aliases the private function `rmi._run_inference_numpy_1_hidden()`. The other possible aliases are `rmi._run_inference_numpy_0_hidden()`, `rmi._run_inference_numpy_2_hidden()` and `rmi._run_inference_numpy_n_hidden()`.

The private function `rmi._run_inference_tensorflow()` can be used to run inference directly using TensorFlow, although it is much (e.g. 200x) slower than the Numpy versions.


## Prerequisites

TensorFlow  
Python 3.x


## License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md) for details


