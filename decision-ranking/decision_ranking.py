# Importing the libraries for matrix manipulation and distance computing
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from numpy.core.function_base import geomspace

### Distance vector building

## Coordinate extracted manually from google map

# Read file locations (multiple locations)
array_locations = np.genfromtxt(r'locations', delimiter=',')

# Convert element in numpy array to tuple and stack into a list
array_locations = list(map(tuple, array_locations))

# Read file her_location
her_location = np.genfromtxt(r'her_location', delimiter=',')

# Convert her location np array to tuple
her_location = tuple(her_location)

# Generate distance vector from her home to several locations
distance = [geodesic(her_location, array_locations[i]).km for i in range (len(array_locations))]
# Generate ranking distance vector
distance = np.array(pd.Series(distance).rank()).astype("float32")
distance = np.reshape(distance,(-1,1))

### Interest vector building

# Set her encode interest to the reference [1, 2, 3, 4, 5] for later comparsion

# Generate her degrees of interest 
her_interest = np.genfromtxt(r'her_interest_encode', delimiter=',')
her_interest = np.reshape(her_interest,(-1,1))


# Generate my degrees of interest
my_interest = np.genfromtxt(r'my_interest_encode', delimiter=',')
my_interest = np.reshape(my_interest,(-1,1))


### Pricing vector building
prices = np.genfromtxt(r'prices', delimiter=',')
# Generate ranking pricing vector
prices = np.array(pd.Series(prices).rank()).astype("float32")
prices= np.reshape(prices,(-1,1))

### Generate matrix of ranking features (distance, her_interest, my_interest, prices)
ranking_matrix = np.concatenate((distance, her_interest, my_interest, prices), axis=1)

### Choose weight vector for 4 features
weight_vector = np.array([[1, 2, 2, 1]]).T

# Score vector computing
print(np.dot(ranking_matrix,weight_vector))
