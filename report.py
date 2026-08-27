"""Turn a solver combo (six local item indices) back into a readable build.

    python report.py 12 34 56 78 90 11
    python report.py --compare        # is the CPU tool's surrogate top-1 the true optimum?
"""

import argparse
import json
import sys

PARTS = ["武器", "頭盔", "鎧甲", "項鍊", "戒指", "鞋子"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idx", nargs="*", type=int, help="six local item indices, solver order")
    ap.add_argument("--compare", action="store_true",
                    help="show the CPU tool's surrogate top list for comparison")
    ap.add_argument("--data", default="data/problem.json")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(args.data, encoding="utf-8") as fh:
        meta = json.load(fh)

    if args.idx:
        if len(args.idx) != 6:
            sys.exit("need exactly six indices")
        print(f"{meta['character']}  ({meta['objective']}, spd_min {meta['spd_min']})")
        uids = []
        for part, i in zip(PARTS, args.idx):
            it = meta["items"][part][i]
            uids.append(it["uid"])
            holder = f"（{it['holder_name']}）" if it.get("holder_name") else ""
            print(f"  {part} #{i:<3} {it['set_zh']}套 Lv{it['item_level']} +{it['enhance']:<2}"
                  f" 主{it['main_stat']:<9} {it['name']}{holder}")
        tool = {tuple(t["uids"]): rank for rank, t in enumerate(meta["cpu_tool_top"], 1)}
        rank = tool.get(tuple(sorted(uids)))
        print(f"  -> {'CPU 代理工具排名 #%d' % rank if rank else '代理工具 top-%d 沒找到這組!' % len(tool)}")

    if args.compare or not args.idx:
        print(f"\nCPU tool (surrogate + rerank, {meta['cpu_tool_seconds']:.1f}s):")
        for rank, t in enumerate(meta["cpu_tool_top"], 1):
            print(f"  #{rank:<2} {t['value']:>12,.1f}")


if __name__ == "__main__":
    main()
