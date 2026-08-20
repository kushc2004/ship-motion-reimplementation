#!/usr/bin/env python3
"""M8--M11 experiments consuming immutable M0--M7 inputs only."""
from __future__ import annotations
import argparse, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("SHIPMOTION_OUTPUT_ROOT", ROOT / "outputs"))
OUT = OUTPUT_ROOT / "m8_m12"; SEED = 42
sys.path.insert(0, str(ROOT / "src"))
from shipmotion.benchmark import _normalize_x, load_windows, prediction_frame, split_arrays
from shipmotion.data.pipeline import MOTION_COLUMNS
from shipmotion.phase2 import MultiTaskTransformer, causal_score, classification_metrics, feature_data, fit_classical, motion_metrics, verify_frozen_inputs
from shipmotion.artifacts import cached_stage_status

def dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2) + "\n")

def ablations():
    """M8: RF is fixed; only features vary and no test result chooses a model."""
    rows=[]; frames=[]
    for dataset in ("BTP-1","BTP-2"):
      for fs in (["F0_motion_only","F1_wave_motion","F2_wave_speed_motion"] if dataset=="BTP-1" else ["F0_motion_only","F1_wave_motion"]):
        d=feature_data(ROOT,dataset,fs); ti,xtr,ytr=split_arrays(d,"train"); vi,xv,yv=split_arrays(d,"validation"); xi,xt,yt=split_arrays(d,"test")
        _,pred,r=fit_classical("Random Forest",_normalize_x(d,xtr),ytr,_normalize_x(d,xv),yv,_normalize_x(d,xt),yt,d["target_scale"])
        f=prediction_frame("Random Forest",dataset,xi,d,pred); f["feature_set"]=fs; frames.append(f)
        rows.append({"dataset":dataset,"feature_set":fs,"features":d["feature_names"],"model":"Random Forest","selection_scope":"fixed hyperparameters; no test tuning","fit_time_s":r["fit_time_s"],"cpu_batch1_p50_ms":r["cpu_batch1_p50_ms"],"cpu_batch1_p95_ms":r["cpu_batch1_p95_ms"],"validation":r["validation"],"test":r["test"],"test_windows":len(xi)})
    pd.concat(frames,ignore_index=True).to_parquet(OUT/"m8_feature_ablation_predictions.parquet",index=False)
    pd.DataFrame([{**{k:v for k,v in r.items() if k not in ("validation","test")},"validation_macro_nrmse":r["validation"]["macro_nrmse"],"macro_nrmse":r["test"]["macro_nrmse"]} for r in rows]).to_csv(OUT/"m8_feature_ablations.csv",index=False)
    dump(rows,OUT/"m8_feature_ablations.json")

def _speed_windows(block, split):
    b=block[block.split==split].sort_values("source_row_index").reset_index(drop=True); ans=[]
    for i in range(0,len(b)-149,10):
      start,end=int(b.loc[i,"source_row_index"]),int(b.loc[i+149,"source_row_index"])
      if end-start==149: ans.append((str(b.loc[i,"run_id"]),split,start,end))
    return ans

def loso():
    """M9: BTP-1 clock-reset runs are independent speed folds, no concatenation."""
    raw=pd.read_parquet(ROOT/"data/splits/BTP-1_raw.parquet"); speeds=raw.groupby("run_id").ship_speed_kn.first().sort_values()
    cols=["wave_m","ship_speed_kn",*MOTION_COLUMNS]; manifests=[]; results=[]; frames=[]
    for held,speed in speeds.items():
      chunks=[]
      for rid,b in raw.groupby("run_id",sort=False):
        # Split positions must be local to the run. source_row_index remains the
        # immutable global/raw identity recorded in the manifest.
        b=b.sort_values("source_row_index").reset_index(drop=True).copy(); n=len(b); b["split"]="purged"
        if rid==held: b["split"]="test"
        else:
          b.loc[:int(.6*n)-1,"split"]="train"; b.loc[int(.6*n)+150:int(.8*n)-1,"split"]="validation"
        chunks.append(b)
      z=pd.concat(chunks,ignore_index=True); records=[]
      for split in ("train","validation","test"):
        for _,b in z.groupby("run_id",sort=False): records.extend(_speed_windows(b,split))
      man=pd.DataFrame(records,columns=["run_id","split","raw_start_index","raw_end_index"])
      manifests.append({"held_out_run":held,"held_out_speed_kn":float(speed),"split_protocol":"held speed test; temporal train/validation on remaining runs; 150-step purge","windows":man.to_dict("records")})
      def arr(split):
        xs=[];ys=[]
        for rid,sp,start,end in records:
          if sp!=split: continue
          b=z[(z.run_id==rid)&(z.source_row_index>=start)&(z.source_row_index<=end)].sort_values("source_row_index")
          if len(b)!=150: raise AssertionError("LOSO window is not a contiguous raw segment")
          xs.append(b.iloc[:100][cols].to_numpy("float32"));ys.append(b.iloc[100:][MOTION_COLUMNS].to_numpy("float32"))
        return np.stack(xs),np.stack(ys)
      xtr,ytr=arr("train");xv,yv=arr("validation");xt,yt=arr("test");mean=xtr.mean((0,1));std=xtr.std((0,1));std[std==0]=1;target_std=ytr.reshape(-1,6).std(0);target_std[target_std==0]=1
      # The OOD protocol evaluates a fixed, deterministic linear baseline.  The
      # primary RF remains frozen; its large multi-output forest is not refit
      # here merely to create a second, resource-dependent benchmark.
      _,pred,r=fit_classical("Ridge",(xtr-mean)/std,ytr,(xv-mean)/std,yv,(xt-mean)/std,yt,target_std)
      persist=np.repeat(xt[:,-1:,2:],50,axis=1); pm=motion_metrics(yt,persist,target_std)
      for dof,pbase in zip(r["test"]["per_dof"],pm["per_dof"]): dof["persistence_skill"]=1-(dof["rmse"]**2)/(pbase["rmse"]**2) if pbase["rmse"] else None
      meta=[]
      for wi,(rid,sp,start,end) in enumerate(records):
        if sp!="test": continue
        times=z[(z.run_id==rid)&(z.source_row_index>=start)&(z.source_row_index<=end)].sort_values("source_row_index").iloc[100:].time_s.to_numpy()
        meta.append({"window_id":f"loso_{held}_{wi}","run_id":rid,"split":sp,"forecast_origin_time_s":float(times[0]-.1),"target_times":times})
      pseudo={"metadata":meta,"y":yt};frames.append(prediction_frame("Ridge","BTP-1",np.arange(len(meta)),pseudo,pred).assign(held_out_speed_kn=float(speed)))
      results.append({"model":"Ridge","held_out_run":held,"held_out_speed_kn":float(speed),"train_runs":sorted(set(z[z.split=="train"].run_id)),"validation_runs":sorted(set(z[z.split=="validation"].run_id)),"train_windows":len(xtr),"validation_windows":len(xv),"test_windows":len(xt),"metrics":r["test"],"fit_time_s":r["fit_time_s"],"cpu_batch1_p50_ms":r["cpu_batch1_p50_ms"],"cpu_batch1_p95_ms":r["cpu_batch1_p95_ms"]})
    dump(manifests,OUT/"m9_speed_ood_window_manifests.json");dump(results,OUT/"m9_speed_ood.json")
    pd.DataFrame([{k:v for k,v in r.items() if k!="metrics"}|{"macro_nrmse":r["metrics"]["macro_nrmse"]} for r in results]).to_csv(OUT/"m9_speed_ood.csv",index=False);pd.concat(frames,ignore_index=True).to_parquet(OUT/"m9_speed_ood_predictions.parquet",index=False)

def proxy_targets(x,y,depth): return np.asarray([causal_score(future[:,2],depth,history[:,3]) for history,future in zip(x,y)],dtype="float32")

def mtt_latency(model,x):
    model.to("cpu").eval();a=torch.from_numpy(x).float()
    for _ in range(20):
      with torch.no_grad(): model(a)
    v=[]
    for _ in range(100):
      st=time.perf_counter()
      with torch.no_grad(): model(a)
      v.append((time.perf_counter()-st)*1000)
    return float(np.percentile(v,50)),float(np.percentile(v,95))

def train_mtt(depth,task,max_epochs=12):
    """M10: legacy-compatible 128/8/4/512 MTT; validation selects epoch only."""
    d=feature_data(ROOT,"BTP-2","F1_wave_motion");ti,xtr,ytr=split_arrays(d,"train");vi,xv,yv=split_arrays(d,"validation");xi,xt,yt=split_arrays(d,"test")
    nxtr,nxv,nxt=_normalize_x(d,xtr),_normalize_x(d,xv),_normalize_x(d,xt);str_,sv,st=proxy_targets(xtr,ytr,depth),proxy_targets(xv,yv,depth),proxy_targets(xt,yt,depth);sm,ss=float(str_.mean()),float(str_.std() or 1)
    score_on,flag_on=task!="single_task",task=="multitask";device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu");torch.manual_seed(SEED);np.random.seed(SEED);model=MultiTaskTransformer(7,score_on,flag_on).to(device)
    tensor=lambda x,y,s:(torch.from_numpy(x).to(device),torch.from_numpy((y-d["target_mean"])/d["target_scale"]).to(device),torch.from_numpy((s-sm)/ss).to(device));tx,ty,ts=tensor(nxtr,ytr,str_);vx,vy,vs=tensor(nxv,yv,sv);xx=torch.from_numpy(nxt).to(device)
    flag=torch.from_numpy((str_>=1).astype("float32")).to(device);pos=int(flag.sum().item());weight=torch.tensor(float((flag.numel()-pos)/pos) if pos else 1.,device=device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5);best=float("inf");state=None;stale=0;history=[];started=time.perf_counter()
    for epoch in range(1,max_epochs+1):
      model.train();opt.zero_grad();mo,sc,fg=model(tx);loss=torch.nn.functional.mse_loss(mo,ty)
      if score_on: loss=loss+.5*torch.nn.functional.mse_loss(sc,ts)
      if flag_on: loss=loss+.3*torch.nn.functional.binary_cross_entropy_with_logits(fg,flag,pos_weight=weight)
      loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();model.eval()
      with torch.no_grad(): vm,vs_pred,_=model(vx);vl=torch.nn.functional.mse_loss(vm,vy)+(0 if not score_on else .5*torch.nn.functional.mse_loss(vs_pred,vs))
      history.append({"epoch":epoch,"train_loss":float(loss.item()),"validation_loss":float(vl.item())})
      if vl.item()<best-1e-7: best=vl.item();state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
      else: stale+=1
      if stale>=4: break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():mo,sc,fg=model(xx)
    pred=mo.cpu().numpy()*d["target_scale"]+d["target_mean"];spred=sc.cpu().numpy()*ss+sm if score_on else None;direct=torch.sigmoid(fg).cpu().numpy() if flag_on else None;derived_score=np.asarray([causal_score(p[:,2],depth,h[:,3]) for h,p in zip(xt,pred)],dtype="float32");derived=(derived_score>=1).astype(float);p50,p95=mtt_latency(model,nxt[:1])
    name=f"m10_{task}_{depth:.1f}m";result={"experiment":name,"task":task,"threshold_m":depth,"threshold_label":"realistic proxy" if depth==.5 else "synthetic-threshold operability proxy","proxy":"causal trailing five-sample heave half-range; no verified propeller geometry","architecture":{"d_model":128,"heads":8,"layers":4,"ff_dim":512,"dropout":.1,"loss_weights":{"motion":1.,"score":.5 if score_on else 0.,"flag":.3 if flag_on else 0.}},"device":str(device),"seed":SEED,"epochs":epoch,"fit_time_s":time.perf_counter()-started,"cpu_batch1_p50_ms":p50,"cpu_batch1_p95_ms":p95,"motion":motion_metrics(yt,pred,d["target_scale"]),"score":None if spred is None else {"rmse":float(np.sqrt(np.mean((spred-st)**2))),"mae":float(np.mean(abs(spred-st))),"correlation":float(np.corrcoef(spred.ravel(),st.ravel())[0,1]) if np.std(spred) and np.std(st) else None},"direct_flag":None if direct is None else classification_metrics(st>=1,direct),"motion_derived_flag":classification_metrics(st>=1,derived)}
    dump(result,OUT/f"{name}_metrics.json");dump(history,OUT/f"{name}_training_history.json");torch.save(model.state_dict(),OUT/f"{name}_model.pt");np.savez_compressed(OUT/f"{name}_predictions.npz",motion=pred,true_motion=yt,score_prediction=np.array([]) if spred is None else spred,true_score=st,direct_probability=np.array([]) if direct is None else direct,derived_probability=derived)
    f=prediction_frame("MultiTask Transformer","BTP-2",xi,d,pred);f["experiment"]=name;f.to_parquet(OUT/f"{name}_motion_predictions.parquet",index=False)
    return result

def phase_analysis():
    p=pd.read_parquet(OUTPUT_ROOT/"primary_benchmark_m37/predictions.parquet");rows=[]
    for (model,dof),g in p[p.split=="test"].groupby(["model","dof"]):
      if dof not in ("heave_m","roll_deg","pitch_deg"): continue
      lags=[]
      for _,w in g.groupby("window_id"):
        w=w.sort_values("horizon_step");a=w.y_true.to_numpy();b=w.y_pred.to_numpy();lags.append(int(np.argmax(np.correlate(b-b.mean(),a-a.mean(),"full"))-(len(a)-1)))
      rows.append({"model":model,"dof":dof,"median_lag_steps":float(np.median(lags)),"p95_abs_lag_steps":float(np.percentile(abs(np.asarray(lags)),95)),"median_lag_s":float(np.median(lags)/10)})
    pd.DataFrame(rows).to_csv(OUT/"m11_phase_lag.csv",index=False);return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerun even if the cached artifact verifies")
    args = parser.parse_args()
    status = cached_stage_status(ROOT, OUTPUT_ROOT, "phase2")
    if status["complete"] and not args.force:
        print(f"Reusing verified M8--M11 artifact: {status['path']}")
        return
    if not status["complete"]:
        print(f"M8--M11 cache unavailable: {status['reason']}; running experiments.")
    OUT.mkdir(parents=True,exist_ok=True);dump(verify_frozen_inputs(ROOT, OUTPUT_ROOT),OUT/"frozen_input_verification_before.json");dump({"python":platform.python_version(),"torch":torch.__version__,"mps_available":torch.backends.mps.is_available()},OUT/"environment.json")
    print("M8 feature ablations",flush=True);ablations();print("M9 speed-held-out evaluation",flush=True);loso();print("M10 multi-task experiments",flush=True);r=[train_mtt(.5,"single_task"),train_mtt(.5,"score"),train_mtt(.5,"multitask"),train_mtt(.2,"multitask")];dump(r,OUT/"m10_multitask_summary.json");print("M11 phase analysis",flush=True);dump(phase_analysis(),OUT/"m11_phase_summary.json");dump(verify_frozen_inputs(ROOT, OUTPUT_ROOT),OUT/"frozen_input_verification_after.json");print("M8--M11 complete",flush=True)

if __name__=="__main__": main()
