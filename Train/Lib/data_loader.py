#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[ ]:


class DataLoader():
    
    
    def __init__(self, path_vid, path_labels, path_train = None , path_val = None, path_test = None):
        self.path_vid = path_vid
        self.path_labels = path_labels
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        
        self.get_labels(path_labels)
        
        if self.path_train:
            self.train_df = self.load_video_labels(self.path_train)
        if self.path_val:
            self.val_df = self.load_video_labels(self.path_val)
        if self.path_test:
            self.test_df = self.load_video_labels(self.path_test, mode="input")
        
        
        
    def get_labels(self, path_labels):
        
        self.labels_df = pd.read_csv(path_labels, names=['label'])
        #extract labels from dataframe
        self.labels = [str(label[0]) for label in self.labels_df.values]
        self.n_labels = len(self.labels)
        #create dictionaries to convert label to int and backwards
        self.label_to_int = dict(zip(self.labels, range(self.n_labels)))
        self.int_to_label = dict(enumerate(self.labels))
        
        
    def load_video_labels(self, path_subset, mode= "label"):
        if mode == "input":
            names=["video_id"]
        elif mode == "label":
            names = ["video_id", "label"]
            
        df = pd.read_csv(path_subset, sep=';', names=names)
        
        if mode == "label":
            df = df[df.label.isin(self.labels)]
            
            #print(df)
                
        return df 
            
            
            
#   def categorical_to_label():

