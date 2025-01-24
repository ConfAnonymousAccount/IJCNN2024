
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from package.data.dataset import DataSet
from package.data.utils import load_dataframe, get_feature_list


class LifeLineDataSet(DataSet):
    def __init__(self, 
                 data_path, 
                 dataset_name, 
                 **features_kwargs):
        super().__init__()
        self.target_variable = features_kwargs.get("target", "long_covid_intensity")
        self.include_date = features_kwargs.get("include_date", False)
        self._features_list = None
        self.import_dataset(data_path, dataset_name, **features_kwargs)


    def import_dataset(self, data_path: str, dataset_name: str="merged_vaccin_only_1_full.csv", **features_kwargs):
        """import the medical dataset

        Parameters
        ----------
        data_path : str
            path to the dataset
        dataset_name : str, optional
            the dataset name, by default "merged_vaccin_only_1_full.csv"
        **features_kwargs : dict
            it include arguments as :
            {"all": False,
             "symptoms": True,
             "vaccination": True,
             "static": True,
             "booster": False,
             "target": "long_covid_intensity",
             "include_date": False}
        """
        parse_dates = ["RESPONSE_DATE", "INFECTION_DATE"]
        self.data = load_dataframe(data_path, dataset_name, parse_dates=parse_dates)
        # vaccin_2 = load_dataframe(data_path, "merged_vaccin_only_2_full.csv", parse_dates=parse_dates)
        # vaccin_3 = load_dataframe(data_path, "merged_vaccin_only_3_full.csv", parse_dates=parse_dates)
        # vaccin_4 = load_dataframe(data_path, "merged_vaccin_only_4_full.csv", parse_dates=parse_dates)

        # rename some variables
        self.data.rename(columns={"income_before_covid_enum": "income", 
                                  "Feeling suddenly warm, then suddenly cold again": "Feeling warm and cold", 
                                  "Numbness or tingling somewhere in your body": "Numbness or tingling", 
                                  "A feeling of heaviness in your arms or legs": "Heaviness in arms or legs", 
                                  'Part of your body feeling limp or heavy': "Feeling limp or heavy", 
                                  "COVIDVACCINE_ADU": "Vaccine"}, 
                          inplace=True)
        # vaccin_2.rename(columns={"income_before_covid_enum": "income", "Feeling suddenly warm, then suddenly cold again": "Feeling warm and cold", "Numbness or tingling somewhere in your body": "Numbness or tingling", "A feeling of heaviness in your arms or legs": "Heaviness in arms or legs", 'Part of your body feeling limp or heavy': "Feeling limp or heavy", "COVIDVACCINE_ADU": "Vaccine"}, inplace=True)
        # vaccin_3.rename(columns={"income_before_covid_enum": "income", "Feeling suddenly warm, then suddenly cold again": "Feeling warm and cold", "Numbness or tingling somewhere in your body": "Numbness or tingling", "A feeling of heaviness in your arms or legs": "Heaviness in arms or legs", 'Part of your body feeling limp or heavy': "Feeling limp or heavy", "COVIDVACCINE_ADU": "Vaccine"}, inplace=True)
        # vaccin_4.rename(columns={"income_before_covid_enum": "income", "Feeling suddenly warm, then suddenly cold again": "Feeling warm and cold", "Numbness or tingling somewhere in your body": "Numbness or tingling", "A feeling of heaviness in your arms or legs": "Heaviness in arms or legs", 'Part of your body feeling limp or heavy': "Feeling limp or heavy", "COVIDVACCINE_ADU": "Vaccine"}, inplace=True)
        
        self._features_list = get_feature_list(**features_kwargs)

        # Extract the selected features
        self.data = self.data[self._features_list].dropna().reset_index(drop=True)
        # vaccin_2_extract = vaccin_2[features_list].dropna().reset_index(drop=True)
        # vaccin_3_extract = vaccin_3[features_list].dropna().reset_index(drop=True)
        # vaccin_4_extract = vaccin_4[features_list].dropna().reset_index(drop=True)

        self.__preprocess_data()

        if self.include_date:
            self.__treat_date()

        # self.vaccin_2 = vaccin_2_extract
        # self.vaccin_3 = vaccin_3_extract
        # self.vaccin_4 = vaccin_4_extract

    def __preprocess_data(self):
        # Remove some minority modalities and reduce the health modalities
        if "Vaccine" in self._features_list:
            self.data = self.data[self.data["Vaccine"]!="only second"]
            self.data = self.data[self.data["Vaccine"]!="only first"]
            self.data = self.data[self.data["Vaccine"]!="Not say"]
            self.data = self.data[self.data["Vaccine"]!="first jab"]
        if "health" in self._features_list:
            self.data.loc[self.data["health"] =="excellent", "health"] = "very good"
            self.data.loc[self.data["health"] =="poor", "health"] = "mediocre"

    def __treat_date(self):
        infection_date = self.data.pop("INFECTION_DATE")
        self.data["INF_YEAR"] = infection_date.dt.year
        self.data["INF_MONTH"] = infection_date.dt.month
        self.data["INF_DAY"] = infection_date.dt.day
        self.data["INF_WEEK_NUM"] = np.asarray(infection_date.dt.isocalendar().week, dtype=int)
    
    def get_encoded_data(self):
        # One hot encoding for all the categorical variables
        features = pd.get_dummies(self.data, dtype=int)
        # Consider long covid as the target
        labels = features.pop(self.target_variable)
        if labels.dtype != float:
            labels = np.asarray(labels, dtype=int)
        # keep the features list for further analysis
        feature_list = list(features.columns)
        # transform the features to array for models
        features = np.array(features)

        self.features = features
        self.targets = np.asarray(labels).reshape(-1,1)
        self.feature_list = feature_list
        self.data_size = len(self.targets)

        #return features, labels, feature_list

    def get_encoded_custom(self):
        
        columns_to_encode = []
        columns_name = ["Vaccine", "GENDER", "variant", "smoking_enum", "chronic_enum"]
        for name_ in columns_name:
            if name_ in self._features_list:
                columns_to_encode.append(name_)
        

        features = pd.get_dummies(self.data, dtype=int, columns=columns_to_encode)
        labels = features.pop(self.target_variable)
        if labels.dtype != float:
            labels = np.asarray(labels, dtype=int)

        if "health" in self._features_list:
            oe = OrdinalEncoder(categories=[["mediocre", "good", "very good"]])
            oe.fit(features["health"].values.reshape(-1,1))
            features["health"] = oe.transform(features["health"].values.reshape(-1,1))
        
        # keep the features list for further analysis
        feature_list = list(features.columns)
        # transform the features to array for models
        features = np.array(features)

        self.features = features
        self.targets = np.asarray(labels).reshape(-1,1)
        self.feature_list = feature_list
        self.data_size = len(self.targets)
