### Catalog
The catalog employed can be found in the zip file 'custom_catalog.csv.zip' and it is divided into 'sequences' (seq_id). This 'custom_catalog' has been obtained from the JMA one [https://www.data.jma.go.jp/svd/eqev/data/bulletin](here) by separating the earthquakes into large mainshocks (of magnitude >= 6) and aftershocks.

### Dataset
The main files for creating a dataset are:
- velset_builder.ipynb
- dset_builder.ipynb
Basically, velset_builder.ipynb takes as input a seismic catalog (in our case 'custom_catalog.csv') by setting the appropriate flag 'catalog_type' to 'custom', loads it and finds all the GPS stations approapriate for each mainshock.

A catalog should be formatted as follows (at the end of any preprocessing):
- Columns: 'date', 'seq_id', 'lat', 'lon', 'mag', 'type'
Where 'seq_id' is a unique sequence identifier and 'type' can be either 1 or 2. 1 for mainshock, 2 for aftershocks (if we were to add foreshocks, we would use 0 as well).
In our version of the code we preprocess two different catalog files. In a future release we will make this more self-consistent.

After running the whole notebook, a file is created in a subfolder 'velset'

This file is used as an input for the dset_builder.ipynb that generates a dataset with all the ground deformation maps and the label for aftershocks forecasting.
The dataset is a Python dictionary containing input and labels as well as any other useful data (such as sequence IDs, day of the MS...).


### Neural Network Model

The CNN employed was made of

Column (1): 5 convolutional layers of 10 channels, kernel size 3, stride 1, padding 'same', padding mode 'replicate'. Between each layer ReLU activation.
Column (2): identical to (1)
Combined: 5 convolutional layers of [10, 10, 5, 5, 1] channels, [3,3,3,3,1] kernel size, stride 1, padding 'same', padding mode 'replicate'. Between each layer ReLU activation part from the last layer, where a Sigmoid is present.

### Monument Table
We also provide the monument table for the GPS stations employed.




