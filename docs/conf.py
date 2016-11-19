import mock

MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate','matplotlib.sphinxext.mathmpl']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
