"""
Compatibility shim: hibs_LMT → hibs_lnn
"""
import sys
import importlib

hb = importlib.import_module('hibs_lnn')
sys.modules['hibs_LMT'] = hb
sys.modules['hibs_LMT.v16_6_50m'] = importlib.import_module('hibs_lnn.v16_6_50m')

for key in list(hb.__dict__.keys()):
    if not key.startswith('_'):
        globals()[key] = getattr(hb, key)
