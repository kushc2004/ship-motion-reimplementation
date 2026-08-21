#!/usr/bin/env python3
"""Export M12 only after frozen M8--M11 artifacts have completed."""
from __future__ import annotations
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUTPUT_ROOT=Path(os.environ.get('SHIPMOTION_OUTPUT_ROOT', ROOT/'outputs'))
PHASE=OUTPUT_ROOT/'m8_m12'; REPORTS=OUTPUT_ROOT/'reports'; FIG=OUTPUT_ROOT/'figures'
def load(path): return json.loads(path.read_text())
def flag_text(x):
    if x is None: return 'not applicable'
    if x.get('status')=='N/A': return f"N/A ({x['reason']}; positives={x['positive_support']}, negatives={x['negative_support']})"
    return 'F1={f1:.4f}, balanced accuracy={balanced_accuracy:.4f}, PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}'.format(**x)
def make_figures(primary,ood,phase):
    FIG.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,4));ax.scatter(primary.cpu_batch1_p50_ms,primary.macro_nrmse)
    for r in primary.itertuples(): ax.annotate(r.model,(r.cpu_batch1_p50_ms,r.macro_nrmse),xytext=(3,3),textcoords='offset points',fontsize=8)
    ax.set(xlabel='CPU batch-1 p50 latency (ms)',ylabel='Macro NRMSE (lower is better)',title='Frozen primary: accuracy vs latency');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'m11_accuracy_latency.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(ood.held_out_speed_kn.astype(str),ood.macro_nrmse);ax.set(xlabel='Held-out vessel speed (kn)',ylabel='Macro NRMSE',title='M9 leave-one-speed-out Ridge');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(FIG/'m9_speed_ood.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for dof,g in phase.groupby('dof'): ax.plot(g.model,g.median_lag_s,marker='o',label=dof)
    ax.axhline(0,color='black',linewidth=.8);ax.set(ylabel='Median cross-correlation lag (s)',title='M11 phase/timing');ax.tick_params(axis='x',rotation=35);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'m11_phase_lag.png',dpi=160);plt.close(fig)
def main():
    required=[PHASE/'frozen_input_verification_after.json',PHASE/'m10_multitask_summary.json',PHASE/'m9_speed_ood.json',PHASE/'m11_phase_lag.csv']
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit('M12 blocked: '+', '.join(missing))
    if load(PHASE/'frozen_input_verification_after.json').get('status') != 'verified': raise SystemExit('M12 blocked: frozen inputs changed')
    primary=pd.read_csv(OUTPUT_ROOT/'primary_benchmark_m37'/'primary_comparison.csv'); comp=load(OUTPUT_ROOT/'primary_benchmark_m37'/'transformer_comparison.json'); ab=pd.read_csv(PHASE/'m8_feature_ablations.csv'); ood=pd.read_csv(PHASE/'m9_speed_ood.csv'); phase=pd.read_csv(PHASE/'m11_phase_lag.csv'); m10=load(PHASE/'m10_multitask_summary.json');make_figures(primary,ood,phase)
    rows=[{'experiment':r['experiment'],'threshold':r['threshold_label'],'motion_macro_nrmse':r['motion']['macro_nrmse'],'fit_time_s':r['fit_time_s'],'cpu_p50_ms':r['cpu_batch1_p50_ms'],'cpu_p95_ms':r['cpu_batch1_p95_ms']} for r in m10]
    single=next(r for r in m10 if r['task']=='single_task' and r['threshold_m']==.5);multi=next(r for r in m10 if r['task']=='multitask' and r['threshold_m']==.5);delta=multi['motion']['macro_nrmse']-single['motion']['macro_nrmse']; verdict='does' if comp['transformer_outperforms'] else 'does not'
    lines=['# Final leakage-free ship-motion benchmark','', '## Frozen protocol','', 'All M8–M11 artifacts consumed the existing immutable M0–M7 split/window inputs. Before/after hash verification passed. All models use 100 history steps, 50 target steps, stride 10, and train-only preprocessing.','', '## Primary motion benchmark','',primary.to_markdown(index=False,floatfmt='.6f'),' ',f"The Transformer **{verdict}** outperform {comp['strongest_non_transformer']} (Macro NRMSE {comp['transformer_macro_nrmse']:.6f} vs {comp['strongest_non_transformer_macro_nrmse']:.6f}).",'', '## M8 feature ablations','',ab[['dataset','feature_set','macro_nrmse','validation_macro_nrmse','cpu_batch1_p50_ms','cpu_batch1_p95_ms']].to_markdown(index=False,floatfmt='.6f'),'', 'The Random Forest was fixed for this ablation; no test metric selected features or hyperparameters.','', '## M9 speed OOD','',ood[['model','held_out_speed_kn','train_windows','validation_windows','test_windows','macro_nrmse']].to_markdown(index=False,floatfmt='.6f'),'', 'Every speed was held out as a full simulation run. The remaining runs use local temporal train/validation blocks with a 150-step purge; window manifests record immutable raw start/end rows. Evidence is limited to one run per speed.','', '## M10 operability proxies and negative transfer','',pd.DataFrame(rows).to_markdown(index=False,floatfmt='.6f'),'', 'No genuine numeric propeller geometry, coordinates, or dimensions were available. This is a causal trailing five-sample heave half-range proxy, not physical propeller emergence.',f"At 0.5 m, multi-task minus single-task motion Macro NRMSE is {delta:+.6f}; a positive value indicates negative transfer.",'- Direct flag head (0.5 m): '+flag_text(multi['direct_flag']),'- Motion-derived flag (0.5 m): '+flag_text(multi['motion_derived_flag']),'- The 0.2 m result is separately labeled as a synthetic stress test.','', '## M11 timing and latency','',phase.to_markdown(index=False,floatfmt='.6f'),'', 'Figures: `outputs/figures/m11_accuracy_latency.png`, `outputs/figures/m9_speed_ood.png`, and `outputs/figures/m11_phase_lag.png`.','', '## Reproduction','', '```bash','PYTHONPATH=src .venv-benchmark311/bin/python scripts/run_m8_m12.py','PYTHONPATH=src .venv-benchmark311/bin/python scripts/export_final_report.py','PYTHONPATH=src .venv-benchmark311/bin/python scripts/export_cv_claims.py','```','', 'No legacy metric was copied into these outputs.']
    REPORTS.mkdir(parents=True,exist_ok=True);(REPORTS/'FINAL_BENCHMARK_REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
