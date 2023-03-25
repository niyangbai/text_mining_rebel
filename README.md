# Text Mining with REBEL Model

### Introduction
This project aims to extract finance-related relations from text using the REBEL model. The REBEL model is a supervised machine learning model that uses a combination of rule-based and statistical approaches for information extraction.

This README file provides an overview of the project and instructions for running the Python scripts included in the project.

### Usage
#### 01_read_data.py
The `01_read_data.py` script takes `TypeSystem.xml`, a repository of `.xmi` files, and a target CSV as arguments. It converts the INCEPTION-labelled `.xmi` files into REBEL-readable `.csv` files.

To run the script, use the following command:

```console
python 01_read_data.py --typesystem <path_to_TypeSystem.xml> --input_dir <path_to_input_directory> --output_file <path_to_output_file>
```

Replace `<path_to_TypeSystem.xml>` with the path to the TypeSystem.xml file, `<path_to_input_directory>` with the path to the directory containing the .xmi files, and `<path_to_output_file>` with the desired path and filename of the output file.

#### 02_training.py

The 02_training.py script trains the REBEL model using the training data set.

#### 03_prediction.py

The 03_prediction.py script predicts the finance-related relations from the test data set using the trained model.