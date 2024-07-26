import importlib
import os
import os.path as osp
# import sys
# sys.path.append('/content/drive/MyDrive/TCL-master')  

from utils.model_register import import_models, Register
# from content.drive.MyDrive.TCL_master.utils.model_register import import_models, Register
print("I am inside models/__init__.py")
model_dict = Register('model_dict')


import_models(osp.dirname(__file__), 'models')
import_models(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'experiment'), 'experiment')
