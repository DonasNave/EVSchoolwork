import source.visualization as vis
import importlib
import inspect
import sys
import os

module_path = os.path.abspath(os.path.join("./source"))
sys.path.append(module_path)

benchmarks = importlib.import_module("benchmark_functions")
functions = [
    obj for name, obj in inspect.getmembers(benchmarks) if inspect.isfunction(obj)
]

for func in functions:
    vis.visualize_2d_function(func)
    vis.visualize_3d_function(func)
