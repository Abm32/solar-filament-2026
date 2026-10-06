"""V27 evaluator: bind four trained checkpoints to the verified V26 engine.

Modes: verify, tune --run NAME [--shard N], merge-run --run NAME,
merge-tune, audit --run NAME --shard N, merge-audit.
No mode reads test images, exports, or submits.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import pandas as pd

BASE=Path(os.environ.get("V27_BASE","/mnt/solar-v27-sliced-finetune")); INPUTS=BASE/"inputs"; RUNTIME=BASE/"runtime"; OUTPUT=BASE/"evaluation"; DATA_ROOT=Path(os.environ.get("V27_DATA_ROOT","/home/azureuser/solar-filament-2026/data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"))
RUNS=("control-seed1","control-seed2","sliced-seed1","sliced-seed2")
RUN_PATHS={run:BASE/f"outputs/train/{run.split('-')[0]}/{run.split('-')[1]}/final.pt" for run in RUNS}
V17_TUNE=0.448702717279432; V17_AUDIT=0.4707634285309831; V26_HYBRID=0.43286880476125505
TUNE_REQUIRED=V17_TUNE+.01; CONTROL_DELTA=.01; V26_DELTA=.02; RAW_MIN=.80; RESCUE_MIN=70; FP_MAX=1.2*306

def engine():
 sys.path.insert(0,str(BASE)); from scripts import evaluate_v26_sliced as v
 v.BASE=BASE;v.DATA_ROOT=DATA_ROOT;v.INPUTS=INPUTS;v.RUNTIME=RUNTIME;v.RAW_IMAGES=DATA_ROOT/"train/train_images";v.ANNOTATIONS=DATA_ROOT/"train/MAGFiLO_1.0_Annotations_kaggle2026_train.json";v.PROVENANCE=INPUTS/"v6_source/v6_provenance.json";v.SOLAR_BASE=INPUTS/"v6_source/solar_base.py";v.CANONICAL_REFINER=INPUTS/"best_refiner_coco.pt";v.V11_REFINER=INPUTS/"proposal_centered_e4.pt";v.HISTORICAL_DIAGNOSTICS=INPUTS/"proposal_diagnostics.csv"
 return v

def configure(run):
 if run not in RUNS:raise ValueError("unknown V27 run")
 v=engine();v.DETECTOR=RUN_PATHS[run];v.DETECTOR_SHA256=v.sha256(RUN_PATHS[run]);v.OUTPUT=OUTPUT/run;return v

def verify():
 data=json.loads((BASE/"dataset_manifest.json").read_text());runs={}
 for run,path in RUN_PATHS.items():
  if not path.is_file():raise RuntimeError(f"missing checkpoint {run}")
  result=json.loads((path.parent/"result.json").read_text()); digest=engine().sha256(path)
  if digest!=result["checkpoint_sha256"]:raise RuntimeError(f"changed checkpoint {run}")
  runs[run]={"checkpoint":str(path),"sha256":digest,"policy":run.split('-')[0],"seed":run.split('-')[1]}
 payload={"status":"verified","dataset_identity":data["dataset_identity"],"runs":runs,"v17_tune":V17_TUNE,"v26_hybrid":V26_HYBRID,"submits":False};engine().write_json(BASE/"evaluation_verification.json",payload);print(json.dumps(payload,indent=2));return payload

def evaluate(run,group,shard):
 v=configure(run)
 if group=="audit":
  selection=json.loads((OUTPUT/"tune_selection.json").read_text())
  if not selection["audit_authorized"]:raise RuntimeError("tune gates did not authorize audit")
  selected_source = (selection["primary"] if run == "sliced-seed1"
                     else selection["matched_control"] if run == "control-seed1"
                     else None)
  if selected_source is None:
   raise RuntimeError("audit is restricted to primary sliced and matched control")
  selected=selected_source["candidate"].replace(f"{run}/","")
  return v.evaluate_names("audit",shard,selected=selected)
 return v.evaluate_names("tune",shard)

def merge_run(run,group="tune"):
 v=configure(run);metrics,raw,proposal,pop=v.merge_group(group)
 hybrids=[key for key in metrics if key.startswith("hybrid/")]
 selected=max(hybrids,key=lambda key:(metrics[key]["pq"],key));_,cell,_,_=v.parse_candidate(selected)
 summary=v.raw_summary(raw,"hybrid",cell["name"]);full=v.raw_summary(raw,"control",cell["name"]);sliced=v.raw_summary(raw,"sliced",cell["name"])
 result={"status":"complete","run":run,"group":group,"population":pop,"candidate":f"{run}/{selected}","metrics":metrics[selected],"raw":summary,"full_raw":full,"sliced_raw":sliced,"proposal_counts":proposal.groupby("cell").sum(numeric_only=True).to_dict("index")}
 v.write_json(OUTPUT/run/f"{group}_selection.json",result);print(json.dumps(result,indent=2));return result

def merge_tune():
 reports={run:json.loads((OUTPUT/run/"tune_selection.json").read_text()) for run in RUNS}
 primary=reports["sliced-seed1"];control=reports["control-seed1"];replica=reports["sliced-seed2"];control_rep=reports["control-seed2"]
 gates={"absolute_tune":primary["metrics"]["pq"]>=TUNE_REQUIRED,"matched_control":primary["metrics"]["pq"]>=control["metrics"]["pq"]+CONTROL_DELTA,"v26_improvement":primary["metrics"]["pq"]>=V26_HYBRID+V26_DELTA,"raw_recall":primary["raw"]["recall"]>=RAW_MIN,"historical_rescue":primary["raw"]["historical_rescued"]>=RESCUE_MIN,"replication_direction":replica["metrics"]["pq"]>control_rep["metrics"]["pq"],"fp_control":primary["metrics"]["fp"]<=FP_MAX or (primary["metrics"]["pq"]>=TUNE_REQUIRED and primary["metrics"]["pq"]>=control["metrics"]["pq"]+CONTROL_DELTA)}
 authorized=all(gates.values());result={"status":"complete","selected_on":"primary sliced seed1 tune PQ","primary":primary,"matched_control":control,"replica":replica,"replica_control":control_rep,"deltas":{"vs_v17":primary["metrics"]["pq"]-V17_TUNE,"vs_control":primary["metrics"]["pq"]-control["metrics"]["pq"],"vs_v26":primary["metrics"]["pq"]-V26_HYBRID,"replica_vs_control":replica["metrics"]["pq"]-control_rep["metrics"]["pq"]},"gates":gates,"audit_authorized":authorized,"action":"evaluate primary and matched control on audit" if authorized else "close V27 without audit/export"}
 engine().write_json(OUTPUT/"tune_selection.json",result);print(json.dumps(result,indent=2));return result

def merge_audit():
 selection=json.loads((OUTPUT/"tune_selection.json").read_text())
 if not selection["audit_authorized"]:raise RuntimeError("tune gates did not authorize audit")
 primary=merge_run("sliced-seed1","audit");control=merge_run("control-seed1","audit")
 raw_delta=primary["raw"]["recall"]-control["raw"]["recall"]
 authorized=primary["metrics"]["pq"]>=V17_AUDIT and primary["metrics"]["pq"]>=control["metrics"]["pq"] and raw_delta>=0
 result={"status":"complete","primary":primary,"matched_control":control,"raw_recall_delta":raw_delta,"required_audit_pq":V17_AUDIT,"export_authorized":authorized,"action":"implement separate test export for explicit authorization" if authorized else "close V27 without export"};engine().write_json(OUTPUT/"v27_evaluation.json",result);print(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("verify","tune","merge-run","merge-tune","audit","merge-audit"));p.add_argument("--run",choices=RUNS);p.add_argument("--shard",type=int,choices=range(4));a=p.parse_args()
 if a.mode=="verify":verify()
 elif a.mode in ("tune","audit"):
  if a.run is None or a.shard is None:p.error("mode requires --run and --shard")
  evaluate(a.run,a.mode,a.shard)
 elif a.mode=="merge-run":
  if a.run is None:p.error("merge-run requires --run")
  merge_run(a.run)
 elif a.mode=="merge-tune":merge_tune()
 else:merge_audit()
if __name__=="__main__":main()
