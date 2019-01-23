## Keras model memory requirements.

- This is an essential python tool to assert the memory requirements of a deep learning model that is implemented utilizing Keras framework and gonna be deployed in TF.
- NN model's size is estimated based on the TF v1.8 specs.

**Module requirements:**

	1. Keras 2.0 or higher
	2. Tensorflow 1.8 or higher 
(I' ll keep you posted if something change regarding memory TF memory mangement)

**How to run**

	- Just hit: "python model_mem_requirements.py" and you will be prompted!
	
	or

	- $ python model_mem_requirements.py -b <batch_size> -p <path/to/model/file> -m <name_of_the_model_definition_in_that_file>

**Theory:**

	total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)

**Repo Focus**

	Similar micro-tools may be provided in the future to constitute a more comprehensive toolkit.
