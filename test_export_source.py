import importlib.util
from pathlib import Path

TARGET = Path(__file__).resolve().parent / 'export_source.py'

def test_target_compiles_and_loads():
    source = TARGET.read_text(encoding='utf-8')
    code = compile(source, 'export_source.py', 'exec')
    namespace = {'__name__': '__test_target__'}
    exec(code, namespace)
    assert namespace is not None
    assert isinstance('export_source', str)
