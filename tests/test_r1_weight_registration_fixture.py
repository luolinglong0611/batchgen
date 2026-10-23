import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/r1_reproduction'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('registration_fixture', SCRIPTS / 'weight_registration_qualification.py')
registration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registration)


def test_shifted_fixture_preserves_bytes_and_forces_real_tensor_crossing(tmp_path):
    original = tmp_path / 'original.bin'
    payload = bytes(range(256)) * (5 * 1024 * 1024 // 256)
    original.write_bytes(payload)
    metadata = {'total_byte_size': len(payload), 'state_dict': {
        'real_tensor': {'offset': 0, 'byte_size': len(payload), 'shape': [len(payload)], 'dtype': 'uint8'}}}
    view = tmp_path / 'view'
    view.mkdir()
    boundary = 8 * 1024**2
    binary, derived, provenance = registration.shifted_fixture(original, metadata, view, boundary)
    prefix = provenance['prefix_bytes']
    assert prefix % 4096 == 0
    assert prefix < boundary < prefix + len(payload)
    with binary.open('rb') as source:
        assert source.read(prefix) == bytes(prefix)
        assert source.read() == payload
    assert provenance['copied_source_sha256'] == hashlib.sha256(payload).hexdigest()
    assert derived['total_byte_size'] == binary.stat().st_size
    assert json.loads((view / 'original.json').read_text()) == derived
    assert metadata['state_dict']['real_tensor']['offset'] == 0
