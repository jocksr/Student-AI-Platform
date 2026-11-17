"""Freeze contenders from both wide-workload protocols before fresh validation."""
import json
from pathlib import Path

from benchmark import save
from sweet_spot import OUT


def main():
    plan=json.loads((OUT/"validation-plan.json").read_text(encoding="utf-8"))
    wide=json.loads((OUT/"wide-range.json").read_text(encoding="utf-8"))["trials"]
    best_long=max((r for r in wide if r["status"]=="complete"),key=lambda r:r["decode_tps"])
    save(OUT/"validation-plan.before-long-winner.json",plan)
    if best_long["config"] not in plan["configurations"]:
        plan["configurations"].insert(3,best_long["config"])
    plan["long_workload_winner"]=best_long
    plan["candidate_note"]="Keep the strongest setting from the two-request wide sweep even when the short recovery probe differed. Compare all contenders on identical fresh prompts; do not rank together unlike workload rates."
    save(OUT/"validation-plan.json",plan)
    print(json.dumps(plan["configurations"],indent=2))


if __name__=="__main__":main()
