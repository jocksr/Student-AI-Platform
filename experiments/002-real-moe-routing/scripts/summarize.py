"""Aggregate observed runs; keep calibration, evaluation and trace data separate."""
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def metrics(run):
    fixed = [r for r in run["requests"] if r["payload"]["ignore_eos"]]
    checked = [r for r in run["requests"] if r["correct"] is not None]
    timing = [r["response"]["timings"] for r in fixed]
    # llama.cpp's decode timer excludes the first output token obtained from
    # prefill, so its denominator is predicted_n - 1.
    steps = sum(t["predicted_n"]-1 for t in timing)
    seconds = sum(t["predicted_ms"] for t in timing)/1000
    return dict(fixed_requests=len(fixed),decode_steps=steps,decode_seconds=seconds,
        decode_tps=steps/seconds if seconds else None,
        mean_ttft_s=statistics.mean(r["first_token_s"] for r in fixed) if fixed else None,
        mean_first_text_s=statistics.mean(r["first_text_s"] for r in fixed) if fixed else None,
        mean_request_s=statistics.mean(r["elapsed_s"] for r in fixed) if fixed else None,
        mean_prefill_s=statistics.mean(t["prompt_ms"] for t in timing)/1000 if timing else None,
        startup_s=run.get("startup_s"),correct=sum(r["correct"] for r in checked),checked=len(checked))


def resources(path):
    rows = [r for r in read(path) if "process_memory" in r]
    return dict(sample_count=len(rows),
        min_available_ram_gib=min(r["system_available_bytes"] for r in rows)/2**30,
        max_process_rss_gib=max(r["process_memory"]["rss"] for r in rows)/2**30,
        max_global_gpu_used_mib=max(r["gpu"].get("used_mib",0) for r in rows),
        max_gpu_temperature_c=max(r["gpu"].get("temperature_c",0) for r in rows)) if rows else {}


def summarize():
    all_runs=[]; failed=[]; evaluation=defaultdict(list); calibration=[]; traces=[]; reasoning=[]
    output_hashes=defaultdict(lambda:defaultdict(list))
    for path in sorted((ROOT/"results/runs").glob("*/run.json")):
        run=read(path); label=run["label"]
        if run.get("status")!="complete":
            failed.append(dict(label=label,status=run.get("status","interrupted"),error=run.get("error"),command=run["command"]))
            continue
        row=dict(label=label,**metrics(run),resources=resources(path.with_name("resource-samples.json")))
        all_runs.append(row)
        if label.startswith("cal-"): calibration.append(row)
        if label.startswith("eval-r"):
            group=label.split("-",2)[2]; evaluation[group].append(row)
            for request in run["requests"]:
                fingerprint=hashlib.sha256(json.dumps(request.get("tokens",[])).encode()).hexdigest()
                output_hashes[request["task_id"]][group].append(fingerprint)
        if label.startswith("trace-"): traces.append(row)
        if label.startswith("reasoning-"):
            reasoning.append(dict(label=label,correct=row["correct"],checked=row["checked"],
                mean_request_s=statistics.mean(r["elapsed_s"] for r in run["requests"]),
                tasks=[dict(task_id=r["task_id"],correct=r["correct"],answer=r["content"],elapsed_s=r["elapsed_s"],
                    predicted_tokens=r["response"]["timings"]["predicted_n"],finish_reason=r["response"]["choices"][0]["finish_reason"])
                    for r in run["requests"]]))
    aggregate=[]
    for name,rows in sorted(evaluation.items()):
        aggregate.append(dict(configuration=name,runs=len(rows),
            decode_tps=sum(r["decode_steps"] for r in rows)/sum(r["decode_seconds"] for r in rows),
            run_decode_tps=[r["decode_tps"] for r in rows],
            median_run_decode_tps=statistics.median(r["decode_tps"] for r in rows),
            mean_request_s=statistics.mean(r["mean_request_s"] for r in rows),
            mean_ttft_s=statistics.mean(r["mean_ttft_s"] for r in rows),
            mean_prefill_s=statistics.mean(r["mean_prefill_s"] for r in rows),
            mean_startup_s=statistics.mean(r["startup_s"] for r in rows),
            correct=sum(r["correct"] for r in rows),checked=sum(r["checked"] for r in rows),
            min_available_ram_gib=min(r["resources"]["min_available_ram_gib"] for r in rows),
            max_process_rss_gib=max(r["resources"]["max_process_rss_gib"] for r in rows),
            max_global_gpu_used_mib=max(r["resources"]["max_global_gpu_used_mib"] for r in rows)))
    trace_pairs=[]
    by_label={r["label"]:r for r in traces}
    for repetition in range(1,4):
        on=by_label.get(f"trace-r{repetition}-on"); off=by_label.get(f"trace-r{repetition}-off")
        if on and off:
            trace_pairs.append(dict(repetition=repetition,trace_on_tps=on["decode_tps"],trace_off_tps=off["decode_tps"],
                decode_time_change_pct=100*(on["decode_seconds"]/off["decode_seconds"]-1),
                request_time_change_pct=100*(on["mean_request_s"]/off["mean_request_s"]-1)))
    consistency={task:dict(unique_hashes_across_configurations=len(set(h for values in configs.values() for h in values)),
        unique_hashes_within_configuration={k:len(set(v)) for k,v in configs.items()}) for task,configs in output_hashes.items()}
    result=dict(evaluation=aggregate,calibration=calibration,tracing=traces,trace_pairs=trace_pairs,
        reasoning_diagnostics=reasoning,output_consistency=consistency,failures=failed,runs=all_runs,
        measurement_notes=["decode_tps is sum(predicted_n - 1) / sum(predicted_ms / 1000), matching this engine's decode timer.",
            "Fixed-length workloads are separate from naturally terminated answer-check workloads.",
            "VRAM is sampled whole-GPU use, including desktop and other programs; RSS is sampled process working set.",
            "Process startup is measured without flushing the operating-system file cache."])
    (ROOT/"results/summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(evaluation=aggregate,calibration=[{k:r[k] for k in ("label","decode_tps","mean_request_s")} for r in calibration],failures=[r["label"] for r in failed]),indent=2))
    return result


if __name__=="__main__": summarize()
