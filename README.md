# openLIS
A Python package that implements a Recursive-Model Index (RMI) and provides a high-level database-like interface.

## Description

This package implements the Recursive-Model Index described in the paper 
"The Case for Learned Index Structures," 
[Kraska et al., 2017](http://arxiv.org/abs/1712.01208)
([pdf](http://arxiv.org/pdf/1712.01208.pdf)).
        
In addition to implementing the Recursive-Model Index, openLIS provides
a high-level database-like interface with Select, Insert, and Delete functionality.

## Getting Started

You can install this package using pip:  

```
pip install openlis
```

### Example usage

For a basic example, see the Jupyter notebook:  

[openlis_example.ipynb](openlis_example.ipynb)

The notebook demonstrates generating a random data set, training the Recursive-Model Index on that data set, and using the three database functions: Select, Insert and Delete.

### Usage details

#### Generating and loading data

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

##### Notes:
* The keys are assumed to be floating point values. If you use something else (like integers), some functions might not work.
* A future implementation should be able to handle different data types.


#### Creating the training DataSet

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
.

#### Create a Recursive-Index-Model

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

The number of linear regression models to use in Stage 2 is set by the input parameter `num_experts`. For example, if you want 100 experts, set

```
num_experts = 100
```
#### Notes
* The number of experts in Stage 2 must be at least 1. 
* Currently not implemented is the possibility of having no Stage 2. 

### Create the database-like interface

Once a model is defined, we can create a database-like interface to that model using

```
rmi_db = openlis.database.IndexStructurePacked(model=rmi)

```

This creates an object of type `IndexStructurePacked`. 

The word "Packed" in `IndexStructurePacked` refers to how the keys are stored. In this initial implementation, the keys are simply stored contiguously in a Numpy array. This means that insertions and deletions will be quite slow, because the array must be resized. 

As proposed in the paper, we could introduce gaps in the stored data, leaving room for key insertions and requiring no additional work after deletions. The resulting class, `IndexStructureGapped`, would share many implementation details with the simpler `IndexStructurePacked`. Thus, the implementation of `IndexStructurePacked` is a logical first step in the implementing the gapped version.

#### Notes
* Need to implement the `IndexStructureGapped` class to speed up insertion and deletion.

### Train!

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

Currently, the user must choose these hyperparameters manually. 

Training proceeds as follows:

* The Stage 1 model is trained, using Mean Squared Error (MSE) as Loss and an Adam optimizer.
*  The Stage 2 model is trained with Stage 1 variables held fixed. Stage 2 also uses MSE and an Adam optimizer.
*  The maximum errors of each linear regression (calculated over the full data set) are calculated and stored.
*  The trained variables are retrieved from the TensorFlow model and stored as Numpy arrays.

### TODO: More sections to come! (e.g. using Select, Insert, and Delete)





### Prerequisites

TensorFlow  
Python 3.x


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


