### Dataset
The main files for creating a dataset are:
- velset_builder.ipynb
- dset_builder.ipynb
Basically, velset_builder.ipynb takes as input a seismic catalog (in our case 'custom_catalog.csv') by setting the appropriate flag 'catalog_type' to either 'custom' or 'giuseppe', loads it and finds all the GPS stations approapriate for each mainshock.

A catalog should be formatted as follows (at the end of any preprocessing):
- Columns: 'date', 'id', 'lat', 'lon', 'mag', 'type'
Where 'id' is a unique sequence identifier and 'type' can be either 1 or 2. 1 for mainshock, 2 for aftershocks (if we were to add foreshocks, we would use 0 as well).
In our version of the code we preprocess two different catalog files. In a future release we will make this more self-consistent.

After running the whole notebook, a file is created in a subfolder 'velset'

This file is used as an input for the dset_builder.ipynb that generates a dataset with all the appropriate useful files.
The dataset is a python dictionary containing input and labels as well as any other useful data (such as sequence IDs, day of the MS...).
