#!/usr/bin/env python3
"""Export project-level CV points from frozen benchmark artifacts."""
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
    if json.loads(files[0].read_text()).get('status') != 'verified': raise SystemExit('CV export blocked: frozen inputs changed')
    m10=json.loads(files[1].read_text())
    multi=next(r for r in m10 if r['task']=='multitask' and r['threshold_m']==.5)
    lines=[
        '# CV-ready project points',
        '',
        'These bullets describe the complete end-to-end ship-motion forecasting project. They are grounded in the frozen benchmark artifacts.',
        '',
        '## Project bullets',
        '',
        '- Developed an end-to-end 6-DoF ship-motion forecasting system that maps a 100-step (10 s) wave-and-motion history to a 50-step (5 s) trajectory forecast.',
        '- Built a simulation-aware maritime data pipeline that discovered independent runs from timestamp resets, retained vessel-speed metadata, and generated run-safe forecasting samples.',
        '- Designed a leakage-free experimental protocol: raw-timeline splits before scaling or windowing, train-only preprocessing, immutable manifests, and automated boundary-overlap tests.',
        '- Implemented and compared persistence, linear-trend, Ridge, Random Forest, XGBoost, LSTM, and Transformer models with identical features, horizons, splits, metrics, and latency measurements.',
        '- Evaluated robustness through feature ablations and leave-one-speed-out testing across 2, 5, and 8 kn simulations, explicitly measuring held-out-speed generalization.',
        '- Developed a multi-task Transformer for motion forecasting and causal operability-proxy scoring; compared motion-only, score-only, direct flag-head, and synthetic stress-test variants.',
        '- Produced reproducible benchmark artifacts with per-DoF and horizon-wise errors, train-normalized Macro NRMSE, persistence skill, phase/timing analysis, predictions, fit time, and CPU inference latency.',
        '',
        '## Evidence boundaries',
        '',
        '- The operability experiments are a heave-based proxy and a separately labelled synthetic stress test, not real propeller-emergence or geometry-aware validation.',
        '- Leave-one-speed-out evidence covers one simulation run at each speed, so it does not establish broad real-world cross-speed generalization.',
    ]
    if multi['direct_flag'].get('status')=='N/A':
        lines.append('- Realistic-threshold classification metrics are N/A because the frozen test split contains only one event class.')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'CV_CLAIMS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
