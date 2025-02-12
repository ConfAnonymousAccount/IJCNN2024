"""
The Scaler class offering the normalization capabilities
"""
from typing import Union
import pathlib
import json
import numpy as np
import copy

#from . import Scaler
#from ...utils import NpEncoder

class StandardScaler(object):
    """Standard scaler

    - X - mean(X) / std(X)
    """
    def __init__(self):
        super().__init__()
        self._m_x = None
        self._m_y = None
        self._std_x = None
        self._std_y = None
        self.indices_to_scale = None

    def fit(self, x, y, indices_to_scale: Union[list, None]=None):
        if indices_to_scale is None:
            self.indices_to_scale = np.arange(x.shape[1])
        else:
            self.indices_to_scale=indices_to_scale

        self._m_x = np.mean(np.asarray(x[:, indices_to_scale], dtype=float), axis=0)
        self._m_y = np.mean(y, axis=0)
        self._std_x = np.std(np.asarray(x[:, indices_to_scale], dtype=float), axis=0)
        self._std_y = np.std(y, axis=0)
        # to avoid division by 0.
        self._std_x[np.abs(self._std_x) <= 1e-6] = 1
        self._std_y[np.abs(self._std_y) <= 1e-6] = 1
        

    def transform(self, x, y):
        x_norm = copy.deepcopy(x)
        y_norm = copy.deepcopy(y)
        x_norm[:, self.indices_to_scale] -= self._m_x
        x_norm[:, self.indices_to_scale] /= self._std_x
        y_norm -= self._m_y
        y_norm /= self._std_y

        return x_norm, y_norm
    
    def transform_x(self, x):
        x_norm = copy.deepcopy(x)
        x_norm[:,self.indices_to_scale] -= self._m_x
        x_norm[:,self.indices_to_scale] /= self._std_x

        return x_norm
    
    def transform_y(self, y):
        y_norm = copy.deepcopy(y)
        y_norm -= self._m_y
        y_norm /= self._std_y

        return y_norm

    def fit_transform(self, x, y):
        self.fit(x, y)
        x_norm = copy.deepcopy(x)
        y_norm = copy.deepcopy(y)
        x_norm[:,self.indices_to_scale] -= self._m_x
        x_norm[:,self.indices_to_scale] /= self._std_x
        y_norm -= self._m_y
        y_norm /= self._std_y

        return x_norm, y_norm

    def inverse_transform(self, y):
        y_origin = copy.deepcopy(y)
        y_origin *= self._std_y
        y_origin += self._m_y
        return y_origin
    
    def inverse_transform_x(self, x):
        x_origin = copy.deepcopy(x)
        x_origin[:,self.indices_to_scale] *= self._std_x
        x_origin[:,self.indices_to_scale] += self._m_x
        return x_origin

    # def save(self, path: Union[str, pathlib.Path]):
    #     """Save the scaler parameters to a file

    #     Parameters
    #     ----------
    #     path : Union[str, pathlib.Path]
    #         the path where the parameters should be saved

    #     Raises
    #     ------
    #     RuntimeError
    #         the fit function is not yet called
    #     """
    #     res_json = {}
    #     res_json["_m_x"] = self._m_x
    #     res_json["_m_y"] = self._m_y
    #     res_json["_std_x"] = self._std_x
    #     res_json["_std_y"] = self._std_y
    #     # for my_attr in ["_m_x", "_m_y", "_std_x", "_std_y"]:
    #     #     tmp = getattr(self, my_attr)
    #     #     if tmp is None:
    #     #         raise RuntimeError(f"The attribute {my_attr} is computed. Call the fit method first.")
    #     #     fun = lambda x: x
    #     #     if isinstance(tmp, np.ndarray):
    #     #         if tmp.dtype == int or tmp.dtype == np.int or tmp.dtype == np.int32 or tmp.dtype == np.int64:
    #     #             fun = int
    #     #         elif tmp.dtype == float or tmp.dtype == np.float32 or tmp.dtype == np.float64:
    #     #             fun = float
    #     #     res_json[my_attr] = [fun(el) for el in tmp]

    #     if not isinstance(path, pathlib.Path):
    #         path = pathlib.Path(path)
    #     if not path.exists():
    #         path.mkdir(parents=True)
    #     with open((path / "scaler_params.json"), "w", encoding="utf-8") as f:
    #         json.dump(obj=res_json, fp=f, indent=4, sort_keys=True, cls=NpEncoder)

    # def load(self, path: Union[str, pathlib.Path]):
    #     """Load the scaler parameters from file

    #     Parameters
    #     ----------
    #     path : Union[``str``, pathlib.Path]
    #         The path to the file where the scaler will be saved.
    #     """
    #     if not isinstance(path, pathlib.Path):
    #         path = pathlib.Path(path)
    #     with open((path / "scaler_params.json"), "r", encoding="utf-8") as f:
    #         res_json = json.load(fp=f)

    #     self._m_x = np.array(res_json["_m_x"], dtype=np.float32)
    #     self._m_y = np.array(res_json["_m_y"], dtype=np.float32)
    #     self._std_x = np.array(res_json["_std_x"], dtype=np.float32)
    #     self._std_y = np.array(res_json["_std_y"], dtype=np.float32)
