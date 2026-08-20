#!/usr/bin/env python3
"""Create conservative claims only from new frozen-test artifacts."""
from __future__ import annotations
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUTPUT_ROOT=Path(os.environ.get('SHIPMOTION_OUTPUT_ROOT', ROOT/'outputs'))
PHASE=OUTPUT_ROOT/'m8_m12';OUT=OUTPUT_ROOT/'cv'
def main():
    files=[PHASE/'frozen_input_verification_after.json',PHASE/'m10_multitask_summary.json',PHASE/'m9_speed_ood.json']
    if not all(p.exists() for p in files): raise SystemExit('CV export blocked until M8–M11 frozen-test artifacts are complete')
    if not json.loads(files[0].read_text()).get('verified'): raise SystemExit('CV export blocked: frozen inputs changed')
    comp=json.loads((OUTPUT_ROOT/'primary_benchmark_m37'/'transformer_comparison.json').read_text());m10=json.loads(files[1].read_text());multi=next(r for r in m10 if r['task']=='multitask' and r['threshold_m']==.5)
    lines=['# CV claims from new frozen-test artifacts only','', '## Defensible','', '- Rebuilt ship-motion evaluation around immutable run-aware raw-timeline splits, train-only preprocessing, and boundary tests.', '- Benchmarked persistence, linear, tree, and neural models on a common 100-step history and 50-step forecast horizon.', '- Ran a three-speed leave-one-speed-out simulation evaluation, holding each speed out as a full run; it is limited to one run per speed.', '- Implemented a legacy-compatible multi-task Transformer and compared single-task, score-only, and direct flag-head variants on frozen test windows.', '- Added a causal heave-based operability proxy and a separately labeled synthetic 0.2 m stress test; neither is physical propeller-emergence validation.', '', '## Do not claim','', '- Do not claim Transformer superiority: the frozen primary comparison found it did not beat the strongest non-Transformer baseline.', '- Do not claim real propeller emergence, geometry-aware operability, real-vessel readiness, or broad cross-speed generalization.', '- Do not reuse legacy notebook/report metrics.']
    if multi['direct_flag'].get('status')=='N/A': lines.insert(10,'- Do not quote realistic-threshold classification quality: the frozen test split has only one event class.')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'CV_CLAIMS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
