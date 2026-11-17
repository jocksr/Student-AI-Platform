"""Publication-style standalone graphs from saved JSON, with no new inference."""
from collections import defaultdict
import json
from pathlib import Path
import os

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"runtime/matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT=ROOT/"results/sweet-spot"
PLOTS=OUT/"graphs"


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def savefig(fig,name):
    fig.savefig(PLOTS/(name+".png"),dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(PLOTS/(name+".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":13,
        "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    aggregate=read(OUT/"curve-aggregates.json")["rows"]
    threads=read(OUT/"curve-thread-choice.json")["threads"]
    selected=read(OUT/"selection.json")
    wide=read(OUT/"wide-range.json")
    wide_plan=read(OUT/"wide-range-plan.json")
    recovery=read(OUT/"recovery-probe.json")
    recovery_aggregate=read(OUT/"recovery-aggregates.json")["rows"]
    trials=read(OUT/"calibration.json")["trials"]+read(OUT/"curve-calibration.json")["trials"]
    repeats=read(OUT/"curve-repeats.json")["trials"]
    by_config=defaultdict(list)
    for row in trials+repeats:
        c=row["config"];by_config[(c["cache_mib"],c["cpu_share"],c["threads"])].append(row)
    # Same two 192-token prompts as tuning: extend the detail view through the
    # actual performance cliff, without mixing in the shorter recovery workload.
    for row in wide["trials"]:
        if row["requested_gib"]<=8:
            c=row["config"];by_config[(c["cache_mib"],c["cpu_share"],c["threads"])].append(row)
    colors={0:"#8c8c8c",.5:"#b68631",.9:"#1769aa",.95:"#169d8f",1.0:"#b34c71"}
    # Keep the explicitly requested 4–32 GiB domain visible, even if the engine
    # cannot allocate all those budgets. Failed cases are not zero-speed data.
    fig,(speed,allocation,outcome)=plt.subplots(3,1,figsize=(13,11),sharex=True,
        gridspec_kw={"height_ratios":[2,1.5,.8]},layout="constrained")
    fig.suptitle("Requested cache budgets: 4–32 GiB",fontsize=18,fontweight="bold")
    speed.set_title("GPT-OSS-20B • RTX 5060 • four CPU threads • two separate workload protocols",loc="left")
    for share in (.9,.95):
        rows=sorted([r for r in wide["trials"] if r["config"]["cpu_share"]==share],key=lambda r:r["requested_gib"])
        xs=[r["requested_gib"] for r in rows]
        ys=[r["decode_tps"] if r["status"]=="complete" else np.nan for r in rows]
        speed.plot(xs,ys,"o-",color=colors[share],label=f"{share:.0%} CPU misses · two 192-token requests",linewidth=2)
        for r in rows:
            x=r["requested_gib"]
            y=share*100
            if r["status"]=="complete":
                status_color="#169d8f" if r["eligible"] else "#c48423"
                outcome.scatter([x],[y],color=status_color,s=55,marker="o")
                if not r["eligible"]:speed.scatter([x],[r["decode_tps"]],s=100,color="#c48423",marker="x",zorder=5)
            else:
                outcome.scatter([x],[y],color="#bf2525",s=60,marker="x")
        attempted=[(r["allocated_cache_mib"] if r["allocated_cache_mib"] is not None else r["failed_allocation_request_mib"])
            for r in rows]
        allocation.plot(xs,[v/1024 if v is not None else np.nan for v in attempted],"o--",color=colors[share],
            label=f"Engine buffer size / failed allocation ({share:.0%})",alpha=.85)
    rx=sorted({r["requested_gib"] for r in recovery["trials"]})
    recovery_by_budget={r["requested_gib"]:r for r in recovery_aggregate}
    ry=[recovery_by_budget[x]["decode_tps"] if x in recovery_by_budget else np.nan for x in rx]
    speed.plot(rx,ry,"s-",color="#7b4a9d",linewidth=2,label="90% CPU misses · one 128-token recovery probe")
    for x in rx:
        row=recovery_by_budget.get(x)
        if row:
            values=row["run_decode_tps"];rate=row["decode_tps"]
            if len(values)>1:speed.errorbar([x],[rate],yerr=[[rate-min(values)],[max(values)-rate]],color="#7b4a9d",capsize=4)
            outcome.scatter([x],[85],color="#7b4a9d",marker="s",s=45)
            if not row["all_complete"]:
                outcome.scatter([x],[85],color="#bf2525",marker="x",s=85,zorder=5)
                speed.scatter([x],[rate],facecolors="none",edgecolors="#bf2525",marker="D",s=90,zorder=5)
        else:outcome.scatter([x],[85],color="#bf2525",marker="x",s=55)
    allocation.plot(rx,[recovery_by_budget[x]["allocated_cache_mib"]/1024 if x in recovery_by_budget else np.nan for x in rx],
        "s:",color="#7b4a9d",label="Recovery-probe allocation",linewidth=1.3)
    speed.set_ylabel("Completed-run decode tokens / second")
    speed.set_ylim(bottom=0);speed.legend(loc="upper right")
    allocation.axhline(wide_plan["expected_full_cache_buffer_mib_from_trace"]/1024,color="#777777",linestyle=":",
        label="Full-expert cache cap from traced tensor sizes")
    allocation.axhline(8151/1024,color="#b68631",linestyle="-.",label="Entire GPU memory; other buffers also need space")
    allocation.set_ylabel("Engine cache-buffer size (GiB)")
    allocation.set_ylim(bottom=0);allocation.legend(loc="lower right",fontsize=8)
    outcome.set_yticks([85,90,95],["90% short probe","90% longer run","95% longer run"])
    outcome.set_ylim(82,99);outcome.set_xlim(3.5,32.5)
    outcome.set_xlabel("Requested GPU expert-cache budget (GiB)")
    outcome.set_xticks([4,5,6,8,10,12,16,24,32])
    outcome.legend(handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor="#169d8f",markeredgecolor="#169d8f",label="Completed, normal headroom"),
        Line2D([0],[0],marker="o",color="none",markerfacecolor="#c48423",markeredgecolor="#c48423",label="Completed, below normal headroom"),
        Line2D([0],[0],marker="x",color="#bf2525",linestyle="none",label="Failed or guard-stopped")],loc="upper center",ncol=3,fontsize=8)
    fig.text(.01,-.04,"A requested budget is not necessarily the allocated cache size. High budgets can hit the full weight-set cap or fail allocation.\nPurple: shorter workload, 1 GiB RAM reserve. Blue/teal: two longer requests, 2 GiB reserve. Bars show repeated probe ranges.\nFailed/stopped trials have no invented throughput. Red diamond: successful probes plotted, but one repeat stopped. Orange: below original headroom target.",fontsize=9)
    savefig(fig,"wide-range")
    fig,(ax,mem)=plt.subplots(2,1,figsize=(12,10),sharex=True,gridspec_kw={"height_ratios":[2,1]},layout="constrained")
    fig.suptitle("GPT-OSS-20B: the useful region and the performance drop",fontsize=17,fontweight="bold")
    ax.set_title(f"Tuning prompts • {threads} CPU threads • RTX 5060 • mmap • 2,048-token context",loc="left")
    failed=[]
    for share,color in colors.items():
        candidates=sorted([(key,values) for key,values in by_config.items() if key[1]==share and key[2]==threads],key=lambda kv:kv[0][0])
        if not candidates:continue
        xs=[];ys=[];mins=[]
        for key,values in candidates:
            good=[r for r in values if r["status"]=="complete"]
            x=key[0]/1024;xs.append(x)
            if not good:
                ys.append(np.nan);mins.append(np.nan);failed.append((x,share));continue
            rate=sum(r["decode_steps"] for r in good)/sum(r["decode_seconds"] for r in good)
            ys.append(rate);mins.append(min(r["min_sampled_gpu_free_mib"] for r in good))
            vals=[r["decode_tps"] for r in good]
            if len(vals)>1:
                ax.errorbar([x],[rate],yerr=[[rate-min(vals)],[max(vals)-rate]],color=color,capsize=4,linewidth=1.2)
            if not all(r["eligible"] for r in values):
                ax.scatter([x],[rate],s=110,marker="x",color="#bf2525",zorder=6)
        label=f"{share:.0%} of cache misses on CPU"
        ax.plot(xs,ys,"o-",color=color,label=label,linewidth=2,markersize=5)
        mem.plot(xs,mins,"o-",color=color,linewidth=1.6,markersize=4)
    ax.set_ylabel("Decode speed (generated tokens / second)")
    ax.set_ylim(bottom=0)
    recommendation=selected["recommended"]["config"]
    chosen_x=recommendation["cache_mib"]/1024
    ax.axvline(chosen_x,color="#343c44",linestyle=":",linewidth=1.2,
        label=f"Selected budget ({recommendation['threads']} threads in validation)")
    mem.axvline(chosen_x,color="#343c44",linestyle=":",linewidth=1.2)
    ax.legend(loc="upper left",fontsize=9,ncol=2)
    mem.axhline(512,color="#bf2525",linestyle="--",linewidth=1.5,label="Original conservative target: 512 MiB free")
    mem.set_ylabel("Minimum sampled GPU free (MiB)")
    mem.set_xlabel("GPU expert-cache budget (GiB)")
    mem.set_ylim(bottom=0);mem.legend(loc="upper right")
    for x in sorted({v[0] for v in failed}):
        shares="/".join(str(round(s*100)) for _,s in sorted(v for v in failed if v[0]==x))
        ax.annotate(f"{shares}% stopped",xy=(x,.02),xycoords=("data","axes fraction"),
            rotation=90,ha="right",va="bottom",fontsize=7,color="#777777")
    mem.set_xlim(.25,8.25)
    mem.set_xticks(np.arange(.5,8.5,.5))
    fig.text(.01,-.025,"Lines connect measured configurations, not a fitted optimum. Bars show repeat min–max where available; single points are one run.\nRed ×: below the original 512 MiB GPU / 3 GiB RAM target. Stopped trials have no invented throughput. Whole-GPU use includes the desktop.",fontsize=9)
    savefig(fig,"cache-curves")

    fig,(ax,ax2)=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    fig.suptitle("CPU settings: more is not automatically faster",fontsize=16,fontweight="bold")
    rows=sorted([r for r in aggregate if r["config"]["cache_mib"]==4096 and r["config"]["cpu_share"]==.95],key=lambda r:r["config"]["threads"])
    ax.plot([r["config"]["threads"] for r in rows],[r["decode_tps"] for r in rows],"o-",color="#1769aa",linewidth=2)
    ax.set(title="4 GiB cache, 95% of misses on CPU",xlabel="CPU threads",ylabel="Decode tokens / second")
    ax.set_xticks([2,4,6,8,10,12,16])
    for cache in sorted({r["config"]["cache_mib"] for r in aggregate if r["config"]["threads"]==threads and r["config"]["cache_mib"]>=3072}):
        rows=sorted([r for r in aggregate if r["config"]["cache_mib"]==cache and r["config"]["threads"]==threads],key=lambda r:r["config"]["cpu_share"])
        if len(rows)<3:continue
        ax2.plot([r["config"]["cpu_share"]*100 for r in rows],[r["decode_tps"] for r in rows],"o-",label=f"{cache/1024:g} GiB")
    ax2.set(title=f"CPU miss share at {threads} threads",xlabel="Cache misses computed on CPU (%)",ylabel="Decode tokens / second")
    ax2.legend(title="Cache budget",fontsize=8,ncol=2)
    fig.text(.01,-.045,"Same tuning prompts. All completed settings shown, including those with insufficient headroom; consult the cache/headroom graph before choosing a setting.",fontsize=9)
    savefig(fig,"cpu-settings")

    rows=[r for r in selected["aggregate"] if "decode_tps" in r]
    labels=[]
    for r in rows:
        c=r["config"]
        labels.append("Static 12 layers\n6 threads" if c.get("static_layers") else f"{c['cache_mib']/1024:g} GiB\n{c['cpu_share']:.0%} CPU misses\n{c['threads']} threads")
    fig,(ax,lat)=plt.subplots(2,1,figsize=(12,10),layout="constrained")
    fig.suptitle("Fresh-prompt validation of the final contenders",fontsize=17,fontweight="bold")
    xs=np.arange(len(rows)); values=[r["decode_tps"] for r in rows]
    fills=["#169d8f" if r["config"]==selected["recommended"]["config"] else "#7896b0" for r in rows]
    ax.bar(xs,values,color=fills,width=.65,alpha=.8)
    for x,r in zip(xs,rows):
        ax.scatter([x]*len(r["run_decode_tps"]),r["run_decode_tps"],color="#152f46",s=26,zorder=4)
        ax.text(x,max(r["run_decode_tps"])+.7,f"{r['decode_tps']:.2f}",ha="center",fontweight="bold")
        if not r["eligible"]:ax.text(x,1,"below original headroom target",ha="center",rotation=90,fontsize=8,color="#bf2525")
    ax.set_xticks(xs,labels,fontsize=9);ax.set_ylabel("Aggregate decode tokens / second")
    ax.axhline(selected["fastest"]["decode_tps"]*.95,color="#b68631",linestyle="--",label="95% of fastest completed cache setting")
    ax.legend(fontsize=8,loc="upper left");ax.set_ylim(0,max(max(r["run_decode_tps"]) for r in rows)+6)
    lat.bar(xs,[r["mean_request_s"] for r in rows],color=fills,width=.65,alpha=.8)
    lat.set_xticks(xs,labels,fontsize=9);lat.set_ylabel("Mean complete 128-token request (seconds)")
    for x,r in zip(xs,rows):lat.text(x,r["mean_request_s"]+.06,f"{r['mean_request_s']:.2f} s",ha="center",fontweight="bold")
    fig.text(.01,-.04,"Three rounds per configuration; four fresh performance prompts per round. Dots are per-round throughput. Green is the selected setting.\nThis graph uses validation prompts; its rates should not be substituted into the tuning curves. Direct-final generation, not full reasoning-enabled chat.",fontsize=9)
    savefig(fig,"validation")
    save_path=PLOTS/"graph-data.json"
    save_path.write_text(json.dumps(dict(curve_threads=threads,curve_aggregates=aggregate,curve_trials=trials+repeats,
        wide_range=wide,recovery=recovery,recovery_aggregates=recovery_aggregate,validation=selected),indent=2)+"\n",encoding="utf-8")
    print("Saved PNG, SVG, and underlying JSON graph data to",PLOTS)


if __name__=="__main__":main()
