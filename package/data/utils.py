import os
import re
import pathlib
from collections import Counter
from tqdm.notebook import tqdm, tqdm_notebook
from typing import Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go

COV_DICT = {'covq_q_t01_results.csv': 'covt01_',
            'covq_q_t02_results.csv': 'covt02_',
            'covq_q_t03_results.csv': 'covt03_',
            'covq_q_t04_results.csv': 'covt04_',
            'covq_q_t05_results.csv': 'covt05_',
            'covq_q_t06_results.csv': 'covt06_',
            'covq_q_t07_results.csv': 'covt07_',
            'covq_q_t08_results.csv': 'covt08_',
            'covq_q_t09_results.csv': 'covt09_',
            'covq_q_t10_results.csv': 'covt10_',
            'covq_q_t11_results.csv': 'covt11_',
            'covq_q_t12_results.csv': 'covt12_',
            'covq_q_t13_results.csv': 'covt13_',
            'covq_q_t14_results.csv': 'covt14_',
            'covq_q_t15_results.csv': 'covt15_',
            'covq_q_t15b_results.csv': 'covt15b_',
            'covq_q_t16_results.csv': 'covt16_',
            'covq_q_t16b_results.csv': 'covt16b_',
            'covq_q_t17_results.csv': 'covt17_',
            'covq_q_t18_results.csv': 'covt18_',
            'covq_q_t19_results.csv': 'covt19_',
            'covq_q_t20_results.csv': 'covt20_',
            'covq_q_t21_results.csv': 'covt21_',
            'covq_q_t22_results.csv': 'covt22_',
            'covq_q_t23_results.csv': 'covt23_',
            'covq_q_t24_results.csv': 'covt24_',
            'covq_q_t25_results.csv': 'covt25_',
            'covq_q_t26_results.csv': 'covt26_',
            'covq_q_t26b_results.csv': 'covt26b_',
            'covq_q_t27_results.csv': 'covt27_',
            'covq_q_t28_results.csv': 'covt28_',
            'covq_q_t29_results.csv': 'covt29_'}

def get_cov_dict():
    return COV_DICT

def get_missing_values_proportion(df):
    """
    This function returns the proportion of missing values per variable

    Parameters
    ----------
        df: ``DataFrame``
            a Pandas dataframe with variables as columns

    Returns
    -------
        ``pd.DataFrame``
            A data frame including the number and proportion of missing values
    """
    df_na = pd.concat([df.isna().sum(), df.isna().sum()/len(df)],axis=1)
    df_na.columns = ['Numbers','Proportions']

    return df_na

def plot_bar_missing_values(df):
    """
    This function trace the bar plot of missing values for the variables of given dataframe

    Parameters
    ----------
        df: ``DataFrame``
            a Pandas dataframe with variables as columns

    Returns
    -------
        None
            Visualizing the barplot of missing values using Plotly library
    """
    df_na = pd.concat([df.isna().sum(), df.isna().sum()/len(df)],axis=1)
    df_na.columns = ['Numbers','Proportions']
    # Plot the missing values proportion
    fig = go.Figure(data=[go.Bar(
                x=df_na.index,
                y=df_na.Proportions,
                text=df_na.Numbers,
                textposition='auto',
            )])
    fig.update_layout(barmode='group',
                    xaxis=dict(
                        tickangle=-45,
                        title='Variables'
                        ),
                    yaxis=dict(
                        title="Proportion"
                        ),
                    title='Rate of missing values per variable')
    fig.show(renderer='notebook')

def read_table(path: Union[pathlib.Path, str], table_name: str):
    """It returns simply a table in its original format

    Parameters
    ----------
    path : Union[pathlib.Path, str]
        Path to where tables could be found
    table_name : str
        the name of the table that should be returned

    Returns
    -------
    ``pd.DataFrame``
        A pandas data frame including the requested table
    """
    covq_table = pd.read_csv(path / table_name)
    return covq_table

def get_var_in_all_tables(path_to_table: Union[pathlib.Path, str],
                          table_list: list,
                          var_name: str,
                          add_prefix: bool = True
                          ):
    """Find a variable in a list of tables

    Parameters
    ----------
    path_to_table : Union[pathlib.Path, str]
        Path to tables where the results are provided
    table_list : ``list``
        A list of table names where the information could be extracted
    var_name : ``str``
        The variable name that should be extracted from the table
    """
    res = list()
    for table_name in table_list:
        covq_table = read_table(path_to_table, table_name)
        covq_table.columns = covq_table.columns.str.upper()
        if add_prefix:
            var_name_in_table = COV_DICT[table_name].upper() + var_name.upper()
        else:
            var_name_in_table = var_name.upper()
        if var_name_in_table in covq_table.columns:
            print("************", var_name_in_table)
            var_to_extract = ["PROJECT_PSEUDO_ID", "VARIANT_ID", "DATE", "AGE", "GENDER", "ZIP_CODE", COV_DICT[table_name].upper() + "responsedate_adu_q_1".upper()]
            new_var_name = var_name_in_table.upper()
            var_to_extract.append(new_var_name)
            tmp_covq_table = covq_table[var_to_extract]
            tmp_covq_table = tmp_covq_table[tmp_covq_table[new_var_name] != 7]
            tmp_covq_table = tmp_covq_table.rename(columns={COV_DICT[table_name].upper() + "responsedate_adu_q_1".upper(): "RESPONSE_DATE"})
            res.append(tmp_covq_table)  
        else:
            res.append(None)
    # concat all the elements in the list to a dataframe
    all_none = sum([el is not None for el in res])
    if all_none > 0:
        res = pd.concat(res)
        res = res.reset_index(drop=True)
    else:
        res = None
    # remove all the nans indicated by $7 in the resulted table
    #res = res[res[new_var_name] != "$7"]
    return res#, new_var_name

def melt_table(df, index: int, var_name: str, value_name:str):
    """Melting table to have variables and values as two columns

    Parameters
    ----------
    df : ``pd.DataFrame``
        _description_
    index : ``int``
        The index from which the variables will be melted
    var_name : ``str``
        the title of the column including the keys
    value_name : ``str``
        the title of the column including the values for the keys in ``var_name``

    Returns
    -------
    ``pd.DataFrame``
        The resulted melted table
    """
    melted_table = pd.melt(df,
                           id_vars=list(df.columns[:index]),
                           var_name=var_name,
                           value_vars=list(df.columns[index:]),
                           value_name=value_name)
    melted_table = melted_table[melted_table[value_name].notna()]
    melted_table = melted_table[melted_table[value_name] != "$7"]
    melted_table[value_name] = [int(el) for el in melted_table[value_name]]
    melted_table = melted_table.sort_values(by=["PROJECT_PSEUDO_ID", "RESPONSE_DATE"]).reset_index(drop=True)

    return melted_table


def get_enum_for_var(table: pd.DataFrame,
                     enumerations: pd.DataFrame,
                     var_name: str,
                     modality_var: str
                     ):
    """_summary_

    Parameters
    ----------
    table : ``pd.DataFrame``
        The table containing a variable for which the enumerations shoud be extracted
    enumerations : ``pd.DataFrame``
        The entire enumeration table including the explaination for each modality of each table
    var_name : ``str``
        The title of column including variable names for which the enumerations should be extracted
    modality_var : str
        The title of the column including the modalities 

    Returns
    -------
    _type_
        _description_
    """
    #enumerations = read_enum_table(path=path_to_enum)
    enums = enumerations[enumerations["VARIABLE_NAME"].str.upper() == var_name]
    enums_en = [enums[enums["ENUMERATION_CODE"] == int(el)]["ENUMERATION_EN"].values[0] for el in table[modality_var]]
    table["ENUMERATION_EN"] = enums_en
    return table

def get_variables_in_tables(path_to_table: str,
                            table_list: list,
                            var_list: list,
                            path_to_enum: str,
                            modality_var: str = "MODALITIES"
                           ):
    """It performs the complete pipeline of extracting a variable list from a table list

    Parameters
    ----------
    path_to_table : ``str``
        Path to the tables where the results (variables) are provided
    table_list : ``list`` of ``str``
        A list containing the table names where the information should be searched
    var_list : ``list`` of ``str``
        A list of variable names that should be extracted in ``table_list``
    path_to_enum : ``str``
        Path to enumeration tables
    modality_var : ``str``, optional
        The column title in which the modalities are provided and for which the enumeration 
        will be extracted, by default "MODALITIES"

    Returns
    -------
    ``pd.DataFrame``
        The resulted data frame containing the variables and their modalities alongside their explanations
    """
    enumerations = read_enum_table(path=path_to_enum)
    
    res_enum_list = []
    for var_ in var_list:
        #print(var_)
        res = get_var_in_all_tables(path_to_table=path_to_table,
                                    table_list=table_list,
                                    var_name=var_)
        if res is not None:
            melted_table = melt_table(res, index=7, var_name="VARIABLE_NAME", value_name=modality_var)
            unique_var_names = melted_table["VARIABLE_NAME"].unique()
            res_enum = get_enum_for_var(table=melted_table,
                                        enumerations=enumerations,
                                        var_name=unique_var_names[0],
                                        modality_var=modality_var
                                        )
            res_enum_list.append(res_enum)
    
    if len(res_enum_list) > 0:
        res_final = pd.concat(res_enum_list).sort_values(by=["PROJECT_PSEUDO_ID", "RESPONSE_DATE"]).reset_index(drop=True)
    else:
        res_final = None
    return res_final
    

def save_dataframe(df, path: Union[str, pathlib.Path], table_name:str):
    df.to_csv(os.path.join(path, table_name + ".csv"))

def load_dataframe(path: Union[str, pathlib.Path],
                   table_name: str,
                   dtype: dict={"PROJECT_PSEUDO_ID": 'string', "AGE": int, "DATE": 'string'},
                   parse_dates: Union[list, str]=['RESPONSE_DATE'],
                   header=0,
                   index_col=0):
    
    df = pd.read_csv(os.path.join(path, table_name),
                     header=header,
                     index_col=index_col,
                     dtype=dtype,
                     parse_dates=parse_dates)
    return df

def get_modality_proportion(df: pd.DataFrame, var_name: str):
    proportions = [el / sum(Counter(df[var_name]).values()) for el in Counter(df[var_name]).values()]
    keys = [el for el in Counter(df[var_name]).keys()]
    df = pd.DataFrame(np.vstack((keys, proportions)).T, columns=["Modalities", "proportions"])
    prop_dict = dict(zip(df.Modalities, df.proportions))
    return prop_dict, df

def get_var_by_id_table(var_name: str, patient_id: list, table_name: str, path: Union[pathlib.Path, str] = None, table: pd.DataFrame = None):
    """
    This function searches for a variable in a specific table for a specific patient
    
    params
    ------
    var_name: ``str``
        the variable name to be extracted
    patient_id: ``list``
        patient identifier in the tables as items is a list
    table_name: ``str``
        a table name in which the variable could be found
    path: ``pathlib.Path``
        a path to the data tables

    Example
    -------
        var_name = "infection_adu_q_1-3"
        patient_id = "4a1e5760-2833-4013-abe6-f188ef608f99"
        table_name = "covq_q_t01_results.csv"
        var = get_var_by_id_table(var_name, patient_id, table_name, table_name)
            
    """
    # read the corresponding tables
    if table is None:
        covq_table = pd.read_csv(path / table_name)
    else:
        covq_table = table
    covq_table.columns = covq_table.columns.str.upper()
    #covq_table = covq_table.set_index("PROJECT_PSEUDO_ID")

    # get the fields with the indicated variable name by keeping the general information
    # concerning the data, patient an other global information that could be used to 
    # merge datasets
    var_name = COV_DICT[table_name].upper() + var_name.upper()
    if var_name in covq_table.columns:
        # filter for the patient id in those tables
        #covq_table = covq_table.loc[patient_id]
        covq_table = covq_table[covq_table["PROJECT_PSEUDO_ID"].isin(patient_id)]
        # covq_table = covq_table.query("PROJECT_PSEUDO_ID == @patient_id") # for a single id
        var_to_extract = ["PROJECT_PSEUDO_ID", "VARIANT_ID", "DATE", "AGE", "GENDER", "ZIP_CODE"]
        var_to_extract.append(var_name.upper())
        res = covq_table[var_to_extract]
    else:
        res = None
    
    return res

def get_list_of_var_by_id_table(var_list: list, patient_id: str, table_name: str, path: Union[pathlib.Path, str], read_table: bool=False):
    """
    This functions searches for a list of variables in a table for a specific patient
    """
    if read_table:
        covq_table = pd.read_csv(path / table_name)
    data_list = list()
    for var_name in var_list:
        if read_table:
            res = get_var_by_id_table(var_name, patient_id, table_name, path, covq_table)
        else:
            res = get_var_by_id_table(var_name, patient_id, table_name, path)
        if res is not None:
            data_list.append(res)
    if len(data_list) == 1:
        return data_list[0]
    return data_list
        

def get_var_by_id_list_of_tables(var_list: list, patient_id: str, table_list: list, path: Union[pathlib.Path, str]):
    """
    This function searches for a variable in a list of tables for a specific patient
    """
    # table_list = [f"covq_q_t{int(el)}_results.csv" for el in table_numbers]
    data_list = list()
    for tbl_name in table_list:
        res = get_list_of_var_by_id_table(var_list, patient_id, tbl_name, path, read_table=True)
        data_list.append(res)
    data_list = [el for el in data_list if type(el) == pd.DataFrame]
    df = pd.concat(data_list).reset_index(drop=True)
    # df.dropna(axis=1, how="all", inplace=True)
    df.iloc[:,6:].dropna(axis=1, how="all", inplace=True)
    return df

def get_var_name_and_tables(variable_table: pd.DataFrame, var_code:str, data_path: pathlib.Path):
    """function to read variable table and extract the variable name and corresponding tables

    Parameters
    ----------
    variable_table : pd.DataFrame
        _description_
    data_path : pathlib.Path
        _description_

    Returns
    -------
    _type_
        _description_
    """
    one_row = variable_table[variable_table["Var"] == var_code]
    var_name = one_row["Code "].values[0]
    var_tables = one_row["Questionnaire"].str.split(",").values[0]
    #table_names = [f"covq_q_t{int(el):02d}_results.csv" for el in var_tables]
    table_names = [f"covq_q_t{el.strip().zfill(2)}_results.csv" for el in var_tables]
    path_to_tables = [data_path / "dataset_order_202201" / "results" / table_name for table_name in table_names]

    return var_name, table_names, path_to_tables

def get_variable_list_by_slash(var_name):
    """This function generate all the variables from a variabe with complex form
    
        for example a variable like infection_adu_q_1/2_a will be decomposed to 
        ``infection_adu_q_1_a`` and ``infection_adu_q_2_a``

    Parameters
    ----------
    var_name : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    first_part = re.search(r'[a-z_]+', var_name).group()
    second_part = re.search(r'([^\d+/]+$)', var_name).group()
    first_number = re.search(r'\d+/\d+', var_name).group()[0]
    second_number = re.search(r'\d+/\d+', var_name).group()[-1]
    var_list = [first_part + str(el) + second_part for el in range(int(first_number), int(second_number)+1)]
    return var_list

def get_variable_list_by_dash(var_name):
    """This function generate all the variables from a variabe with complex form
    
        for example a variable like infection_adu_q_1-2_a will be decomposed to 
        ``infection_adu_q_1_a`` and ``infection_adu_q_2_a``

    Parameters
    ----------
    var_name : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    first_part = re.search(r'[a-z_]+', var_name).group()
    second_part = re.search(r'([^\d+/]+$)', var_name).group()
    first_number = re.search(r'\d+-\d+', var_name).group()[0]
    second_number = re.search(r'\d+-\d+', var_name).group()[-1]
    var_list = [first_part + str(el) + second_part for el in range(int(first_number), int(second_number)+1)]
    return var_list

def get_variable_with_enum(patient_id: list, 
                           variables: pd.DataFrame, 
                           variable_code: str, 
                           sep: str, 
                           variable_aggregated_name:str, 
                           data_path: pathlib.Path, 
                           cov_table_path: pathlib.Path, 
                           enum_table_path: pathlib.Path):
    
    var_name, table_names, table_path  = get_var_name_and_tables(variables, variable_code, data_path)
    if sep == "-":
        var_list = get_variable_list_by_dash(var_name)
    elif sep == "/":
        var_list = get_variable_list_by_slash(var_name)
    elif sep == None:
        var_list = var_name
    else:
        raise ValueError("The corresponding function for this separator is not yet implemented")
    
    res_list = get_var_by_id_list_of_tables(var_list=var_list,
                                            patient_id=patient_id,
                                            table_list=table_names,
                                            path=cov_table_path
                                            )
    # melting the resulted table to have a unique variable representing the desired variable
    melted_list = pd.melt(res_list, 
                          id_vars=list(res_list.columns[:6]), 
                          var_name="TABLE", 
                          value_vars=list(res_list.columns[6:]), 
                          value_name=variable_aggregated_name)
    melted_list = melted_list[melted_list[variable_aggregated_name].notna()]
    melted_list = melted_list.sort_values(by=["PROJECT_PSEUDO_ID", "DATE"]).reset_index(drop=True)
    
    patient_id = melted_list["PROJECT_PSEUDO_ID"].unique()
    res_enum_list = list()
    with tqdm(total=len(patient_id)) as pbar:
        for id_ in patient_id:
            extracted_enum = extract_enum_v3(melted_list.query("PROJECT_PSEUDO_ID == @id_"), variable_name=variable_aggregated_name, path=enum_table_path)
            res_enum_list.append(extracted_enum)
            pbar.update(1)
    
    final_df = pd.concat(res_enum_list).reset_index(drop=True)

    return final_df

def read_enum_table(path: str):
    indices = [index_ for index_, el in enumerate(os.listdir(path)) if "covq_" in el]
    enumeration_tables = np.sort(np.array(os.listdir(path), dtype=str)[indices])

    for id_, table_ in enumerate(enumeration_tables):
        enumeration_content = pd.read_csv(path / table_)
        enumeration_content.columns = enumeration_content.columns.str.upper()
        if id_ == 0:
            enumerations = enumeration_content
        else:
            enumerations = pd.concat([enumerations, enumeration_content])
    enumerations.reset_index(drop=True, inplace=True)
    return enumerations

def extract_enum_v3(df, value_name, variable_name: str = "VARIABLE_NAME", path:str =None):
    """The function to extract the enumeration for a table and unify the explaination

    Parameters
    ----------
    df : ``pd.DataFrame``
        The dataframe issued from vaccination
    variable_name : ``str``
        the variable name on which the extraction should be done 
    path : ``str``
        The path to the enumeration

    Returns
    -------
    ``pd.DataFrame``
        A dataframe with improved enumeration (explaination instead of code)
    """
    indices = [index_ for index_, el in enumerate(os.listdir(path)) if "covq_" in el]
    enumeration_tables = np.sort(np.array(os.listdir(path), dtype=str)[indices])

    for id_, table_ in enumerate(enumeration_tables):
        enumeration_content = pd.read_csv(path / table_)
        enumeration_content.columns = enumeration_content.columns.str.upper()
        if id_ == 0:
            enumerations = enumeration_content
        else:
            enumerations = pd.concat([enumerations, enumeration_content])
    enumerations.reset_index(drop=True, inplace=True)

    data_list = list()
    for var_name in df[variable_name].unique():
        enumeration_extract = enumerations[enumerations["VARIABLE_NAME"]== var_name.lower()]
        df_extract_codes = df[df[variable_name]==var_name][value_name]
        modalities = df_extract_codes.values
        #enumeration_extract[enumeration_extract["ENUMERATION_CODE"] == df_extract_codes.values]
        # TODO: a better and more optimized way should exist to do this

        for el in modalities:
            indices = enumeration_extract["ENUMERATION_CODE"].astype("str")==el
            if len(indices)>0:
                modalities_en = enumeration_extract[indices]["ENUMERATION_EN"].values[0]
            else:
                modalities_en = "NaN"

        #modalities_en = [enumeration_extract[enumeration_extract["ENUMERATION_CODE"].astype("str")==el]["ENUMERATION_EN"].values[0] for el in modalities]
        #modalities_en = enumeration_extract.loc[enumeration_extract.apply(lambda x: x.ENUMERATION_CODE, axis=1)]
        df_extract = df[df[variable_name]==var_name]
        df_extract.loc[:,"ENUMERATION_EN"] = modalities_en
        data_list.append(df_extract) 

    dataframe = pd.concat(data_list).sort_values(by=["PROJECT_PSEUDO_ID", "DATE"]).reset_index(drop=True)
    return dataframe

def extract_enum_v2(df: pd.DataFrame, variable_name: str, path: str) -> pd.DataFrame:
    """_summary_

    TODO: a table with all the modalities and their explaination over all enum tables should be constructed and 
          only search in this table to obtain the explaination for each modality. It will improve the performance

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    path : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    var_names = list(df["TABLE"])
    data_list = list()
    for idx, var_name in enumerate(var_names):
        tbl_num = re.search(r'\d+', var_name).group()
        tbl_name = f"covq_q_t{int(tbl_num):02d}_enumerations.csv"
        covq_enum = pd.read_csv(path / tbl_name)
        covq_enum.columns = covq_enum.columns.str.upper()
        variables_to_extract = ["VARIABLE_NAME", "ENUMERATION_CODE", "ENUMERATION_EN"]
        enum_code = df[df["TABLE"]==var_name][variable_name]
        var_name = var_name.lower()
        res_query = covq_enum.query("VARIABLE_NAME == @var_name")[variables_to_extract]
        res = res_query[res_query["ENUMERATION_CODE"].astype(str) == str(enum_code.values[0]).strip()]
        if res.empty:
            nan_row = pd.DataFrame.from_dict({"VARIABLE_NAME": [np.NaN], "ENUMERATION_CODE": [np.NaN], "ENUMERATION_EN": [np.NaN]})
            res = pd.concat([res, nan_row], axis=0)
        data_list.append(res)
    res_df = pd.concat(data_list).reset_index(drop=True)
    df = df.reset_index(drop=True)
    res_df = pd.concat([df, res_df["ENUMERATION_EN"]], axis=1)
    return res_df

def extract_enum(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Extract corresponding description of each variable modality in df

    TODO: TO be removed in future iterations. Deprecated
          Use the ``extract_enum_v2``version
    """
    # table_list = [f"covq_q_t{int(el)}_enumerations.csv" for el in table_numbers]
    var_names = list(df.columns)[6:]
    data_list = list()
    complete_enum_list = list()
    for idx, var_name in enumerate(var_names):
        tbl_num = re.search(r'\d+', var_name).group()
        tbl_name = f"covq_q_t{int(tbl_num)}_enumerations.csv"
        covq_enum = pd.read_csv(path / tbl_name)
        covq_enum.columns = covq_enum.columns.str.upper()
        variables_to_extract = ["VARIABLE_NAME", "ENUMERATION_CODE", "ENUMERATION_EN"]
        #var_index = [i for i, s in enumerate(var_names) if str(int(tbl_num)) in s][0]
        #var_name = var_names[var_index].lower()
        #print(var_name)
        enum_code = df[df[var_name].notna()][var_name]
        var_name = var_name.lower()
        res_query = covq_enum.query("VARIABLE_NAME == @var_name")[variables_to_extract]
        res = res_query[res_query["ENUMERATION_CODE"] == int(enum_code)]
        data_list.append(res)
        complete_enum_list.append(res_query)
    res_df = pd.concat(data_list).reset_index(drop=True)
    res_complete = pd.concat(complete_enum_list).reset_index(drop=True)
    return res_df, res_complete

def get_feature_list(all=False, 
                     symptoms=False, 
                     static=False, 
                     vaccination=False, 
                     booster=False, 
                     target: str="long_covid", 
                     include_date: bool=True):
    """This function gets the required features name from different categories
        Symptoms
        Static features (age, sex, etc.)
        Vaccination 
        Booster vaccin related features

    Parameters
    ----------
    all : bool, optional
        _description_, by default False
    symptoms : bool, optional
        _description_, by default False
    static : bool, optional
        _description_, by default False
    vaccination : bool, optional
        _description_, by default False
    booster : bool, optional
        _description_, by default False
    target : str, optional
        Indicate which target variable to use for regression or classification, by default "long_covid"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    Exception
        _description_
    """
    generic = ['TIME_TO_INFECTION']
    if include_date:
        generic.extend(["INFECTION_DATE"])
    generic.extend([target])
    vaccination_list = ['Vaccine', 'VACCIN_TTI']
    if booster == True:
        vaccination_list = ['Booster', 'BOOSTER_TTI']
    symptoms_list = ['Headache', 'Dizziness',
       'Heart or chest pain', 'Lower back pain', 'Nausea or upset stomach',
       'Muscle pain/aches', 'Difficulty breathing', 'Feeling warm and cold',
       'Numbness or tingling', 'A lump in your throat',
       'Feeling limp or heavy', 'Heaviness in arms or legs',
       'Pain when breathing', 'Runny nose', 'Sore throat', 'Dry cough',
       'Wet cough', 'Fever (38 degrees or higher)', 'Diarrhea', 'Stomach pain',
       'Loss of sense of smell or taste', 'Red, painful or itchy eyes',
       'Sneezing']
    features_static_list = ["AGE", "GENDER", "variant", "smoking_enum", "chronic_enum", "health"]#, "income"]
    feature_list = []
    if (all == True) or (symptoms == True and static == True and vaccination == True):
        feature_list.extend(vaccination_list)
        feature_list.extend(symptoms_list)
        feature_list.extend(features_static_list)
        feature_list.extend(generic)
    elif (symptoms == True) and (vaccination == True):
        feature_list.extend(symptoms_list)
        feature_list.extend(vaccination_list)
        feature_list.extend(generic)
    elif (symptoms == True) and (static == True):
        feature_list.extend(symptoms_list)
        feature_list.extend(features_static_list)
        feature_list.extend(generic)
    elif (vaccination == True) and (static == True):
        feature_list.extend(vaccination_list)
        feature_list.extend(features_static_list)
        feature_list.extend(generic)
    elif (vaccination == True):
        feature_list.extend(vaccination_list)
        feature_list.extend(generic)
    elif (static == True):
        feature_list.extend(features_static_list)
        feature_list.extend(generic)
    elif (symptoms == True):
        feature_list.extend(symptoms_list)
        feature_list.extend(generic)
    else:
        raise Exception("At least one of the booleans should be True")
    
    return feature_list

def round_off_rating(number):
    """Round a number to the closest half integer.
    >>> round_off_rating(1.3)
    1.5
    >>> round_off_rating(2.6)
    2.5
    >>> round_off_rating(3.0)
    3.0
    >>> round_off_rating(4.1)
    4.0"""

    return round(number * 2) / 2