# cython_loader.py
import os
from setuptools import setup, Extension
import importlib
import numpy


def load_cython_ucsv_functions():
    # 컴파일 후 모듈 import
    module_name = "bok_da.ts.ucsv.ucsv_functions_cython"
    if importlib.util.find_spec(module_name):
        print(f"> Attempting to dynamically import: {module_name}...")
        example = importlib.import_module(module_name)
        return example
    else:
        print(f"Warning: {module_name} not found.")
        return None

def load_cython_ucsv_multivar_functions():
    # 컴파일 후 모듈 import
    module_name = "bok_da.ts.ucsv.ucsv_functions_cython_multivar"
    if importlib.util.find_spec(module_name):
        print(f"> Attempting to dynamically import: {module_name}...")
        example = importlib.import_module(module_name)
        return example
    else:
        print(f"Warning: {module_name} not found.")
        return None



