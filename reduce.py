#!/usr/bin/env python
# coding: utf-8
__author__ = "Tim Holbrook"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Tim Holbrook"


# In[1]:



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, re, glob

for file in glob.glob(os.path.join("C:/Users/ta/Desktop/test_files","*.xlsx")):
# Importing the dataset
    dataset = pd.read_excel(file)
    valueObj = dataset.iloc[16:, 1].values
    
    
    # Create a function called "chunks" with two arguments, l and n:
    def chunks(l, n):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i+n]
    
    test = list(chunks(valueObj,4))
    # takes the max of the the n(the size of the chunked data)
    #this is setup to reduce 0.5 ms data to simulated 2ms dwell time for further processing 
    max = [i.max() for i in test]
    df = pd.DataFrame({'max':max})
    
    #writes the files out with the new name to the given dir 
    new_filename = os.path.join("C:/Users/ta/Desktop/output_files","new_"+os.path.basename(file))
    df.to_excel(new_filename, na_rep='NaN')

