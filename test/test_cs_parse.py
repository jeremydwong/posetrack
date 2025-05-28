# for now, we have left test suites within the actual cs_parse.py file.

# run cs_parse as main to test the parsing
import os
import sys
path_to_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path_to_project_root)

# call cs_parse.py __main__ function
# to run the test suite

if __name__ == "__main__":
    import runpy
    runpy.run_module("src.cs_parse", run_name="__main__")