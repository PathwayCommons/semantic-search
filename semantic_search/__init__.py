import os
import platform

# https://github.com/dmlc/xgboost/issues/1715#issuecomment-420305786
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

__version__ = "0.1.0"
