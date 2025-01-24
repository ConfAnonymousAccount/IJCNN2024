import numpy as np
import pandas as pd
import os

data_path = "/groups/umcg-lifelines/tmp01/projects/ov21_0402/data"
list_dir = os.listdir(data_path)
list_dir.sort()
print(list_dir)
print("*********")

# without hospitalization for the moment
cov_dir = [d for d in list_dir if "hos" not in d]
print(cov_dir)
print("*********")

# take the structure of each directory
list_subdir = os.listdir(os.path.join(data_path, list_dir[0]))
print(list_subdir)
print("*********")

# for each directory in cov_dir
#   do the following
# get the list of files in results
files_path = os.path.join(data_path, cov_dir[0], "results")
files_list = os.listdir(files_path)
files_list = [f for f in files_list if "summary" not in f]
files_list.sort()
print(files_list)
print("*********")

path_to_file = os.path.join(files_path, files_list[0])
dataset_1 = pd.read_csv(path_to_file, index_col=0, low_memory=False)
dataset_1.DATE = pd.to_datetime(dataset_1.DATE)
path_to_file = os.path.join(files_path, files_list[1])
dataset_2 =  pd.read_csv(path_to_file, index_col=0, low_memory=False)
dataset_2.DATE = pd.to_datetime(dataset_2.DATE)

# preparing for merge
# $ drop_list = ['VARIANT_ID', 'DATE', 'AGE', 'GENDER', 'ZIP_CODE'] # 'DATE'
# dropping the repeated columns (attention: maybe date or variant id could be useful. I don't know!)
# if the `Date` is useful, it takes another temporal dimension which is more complex
# In that case, the merge should be done on ID and Date
# Generally, we should know if some tables contain the same variables just in another temporal horizon
# If it is the case, we should identify the tables which contain the same variables and the tables containing 
# differnet variables and to adapt the merge consequently
# $ dataset_2_for_merge = dataset_2.drop(drop_list, axis=1)

# merge
dataset_merged = dataset_1.merge(dataset_2, how="outer", on=["PROJECT_PSEUDO_ID", "VARIANT_ID", "DATE", "AGE", "GENDER", "ZIP_CODE"])

# there are some common variables in these two datasets
# that could correspond to another batch of questions at another date
[el1 for el1 in dataset_1.columns for el2 in dataset_2.columns if el1==el2]

# run in existing python session 
# $ exec(open('read_data.py').read())
