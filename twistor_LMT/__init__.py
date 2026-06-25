"""
Compatibility shim: twistor_LMT → hibs_lnn
"""
import sys
import importlib

hb = importlib.import_module('hibs_lnn')
sys.modules['twistor_LMT'] = hb
sys.modules['twistor_LMT.core'] = importlib.import_module('hibs_lnn.core')
sys.modules['twistor_LMT.growable'] = importlib.import_module('hibs_lnn.growable')
sys.modules['twistor_LMT.datasets'] = importlib.import_module('hibs_lnn.datasets')
sys.modules['twistor_LMT.agent'] = importlib.import_module('hibs_lnn.agent')
sys.modules['twistor_LMT.decoder'] = importlib.import_module('hibs_lnn.decoder')
sys.modules['twistor_LMT.integrators'] = importlib.import_module('hibs_lnn.integrators')
sys.modules['twistor_LMT.ode_solver'] = importlib.import_module('hibs_lnn.ode_solver')
sys.modules['twistor_LMT.analysis'] = importlib.import_module('hibs_lnn.analysis')
sys.modules['twistor_LMT.visualization'] = importlib.import_module('hibs_lnn.visualization')
sys.modules['twistor_LMT.training'] = importlib.import_module('hibs_lnn.training')
sys.modules['twistor_LMT.mobius'] = importlib.import_module('hibs_lnn.mobius')
sys.modules['twistor_LMT.resonance'] = importlib.import_module('hibs_lnn.resonance')
sys.modules['twistor_LMT.self_feedback'] = importlib.import_module('hibs_lnn.self_feedback')
sys.modules['twistor_LMT.reflection'] = importlib.import_module('hibs_lnn.reflection')
sys.modules['twistor_LMT.curiosity_module'] = importlib.import_module('hibs_lnn.curiosity_module')
sys.modules['twistor_LMT.intervention_executor'] = importlib.import_module('hibs_lnn.intervention_executor')
sys.modules['twistor_LMT.episodic_memory'] = importlib.import_module('hibs_lnn.episodic_memory')
sys.modules['twistor_LMT.offline_simulator'] = importlib.import_module('hibs_lnn.offline_simulator')

for key in list(hb.__dict__.keys()):
    if not key.startswith('_'):
        globals()[key] = getattr(hb, key)
