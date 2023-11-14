from emutools.tex import DummyTexDoc
from emutools.utils import load_param_info
from aust_covid.model import build_model
from aust_covid.inputs import get_ifrs


def test_smoke_model():
    """Simple smoke test for epi model.
    """
    param_info = load_param_info()
    dummy_doc = DummyTexDoc()
    parameters = param_info['value'].to_dict()
    parameters.update(get_ifrs(dummy_doc))
    aust_model = build_model(dummy_doc, param_info['abbreviations'])
    aust_model.run(parameters=parameters)
