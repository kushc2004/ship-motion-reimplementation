import sys
from pathlib import Path
import json
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import numpy as np
from shipmotion.phase2 import FEATURE_SETS, MultiTaskTransformer, causal_score, classification_metrics

def test_proxy_is_causal_and_not_geometry():
    a=np.array([0.,0.,0.,0.,10.],dtype=np.float32)
    x=causal_score(a,.5)
    assert x[0] == 0 and x[3] == 0 and x[-1] == 10.
    assert 'ship_speed_kn' in FEATURE_SETS['F2_wave_speed_motion']

def test_single_class_metrics_are_na():
    got=classification_metrics(np.zeros((2,3)),np.zeros((2,3)))
    assert got['status']=='N/A'

def test_multitask_shapes():
    import torch
    m=MultiTaskTransformer(7,True,True); motion, score, flag=m(torch.randn(2,100,7))
    assert motion.shape==(2,50,6) and score.shape==(2,50) and flag.shape==(2,50)

def test_speed_ood_manifest_has_no_cross_split_raw_coverage():
    path=Path(__file__).parents[1]/'outputs'/'m8_m12'/'m9_speed_ood_window_manifests.json'
    assert path.exists(), 'run M9 before asserting its persisted raw-timestep manifest'
    for fold in json.loads(path.read_text()):
        coverage={s:set() for s in ('train','validation','test')}
        for w in fold['windows']:
            coverage[w['split']].update((w['run_id'],i) for i in range(w['raw_start_index'],w['raw_end_index']+1))
            assert w['raw_end_index']-w['raw_start_index']==149
        assert not coverage['train'] & coverage['validation']
        assert not coverage['train'] & coverage['test']
        assert not coverage['validation'] & coverage['test']
