""" This module returns the image and their corresponding mask path """

import numpy as np 
import os
import pandas as pd 
from sklearn.model_selection import train_test_split 


def get_df(img_folder, mask_folder):
    """ Function to return the image and their corresponding mask path in pandas dataframe 
    
    Parameters
    ----------
    img_folder : str
        path of image folder

    mask_folder : str
        path of mask folder

    Returns
    -------
    pandas dataframe
    """

    if os.path.exists(img_folder) and os.path.exists(mask_folder):
        img_ids = []
        img_path = []
        for dirname, _, filenames in os.walk(img_folder):
            for filename in filenames:
                path = os.path.join(dirname, filename)    
                img_path.append(path)
                
                img_id = filename.split(".")[0]
                img_ids.append(img_id)

        d = {"id": img_ids, "img_path": img_path}
        img_df = pd.DataFrame(data = d)
        img_df = img_df.set_index('id')

        mask_ids = []
        mask_path = []
        for dirname, _, filenames in os.walk(mask_folder):
            for filename in filenames:
                path = os.path.join(dirname, filename)
                mask_path.append(path)
                
                mask_id = filename.split(".")[0]
                mask_ids.append(mask_id)

                
        d = {"id": mask_ids,"mask_path": mask_path}
        mask_df = pd.DataFrame(data = d)
        mask_df = mask_df.set_index('id')

        df = pd.merge(img_df, mask_df, on = "id", how = "inner")
        df = df.sort_index()

        return df

    else:
        raise NameError("Folder does not exists")

def split_df(df, test_size=.15, random_state=None):
    ''' Function to split dataframe in train and test dataframe 
    
    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    train_df: pandas dataframe
        image path with corresponding mask path

    test_df: pandas dataframe
        image path with corresponding mask path
    '''
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    return train_df, test_df



