#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[ ]:


def mkdirs(folder, permission):
    if not os.path.exists(folder):
        try:
            original_umask = os.umask(0)
            os.makedirs(folder, permission, exist_ok=True)
        finally:
            os.umask(original_umask)

