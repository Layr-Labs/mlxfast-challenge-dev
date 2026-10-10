#!/usr/bin/env python3
"""Parse the benchmark matrix log into a machine-readable table."""
import json
import re
import sys

log = open(sys.argv[1] if len(sys.argv) > 1 else "matrix.log").read()
blocks = re.findall(r"########## BEGIN (\S+) :: (.*?) ##########\n(.*?)########## END \1 ##########",
                    log, re.S)

rows = []
for cell_id, args, body in blocks:
    r = {"cell": cell_id, "args": args.strip()}

    m = re.search(r"^tokens: (\d+)\s+depth: (\d+)", body, re.M)
    r["tokens"] = int(m.group(1)) if m else None
    r["depth"] = int(m.group(2)) if m else None

    m = re.search(r"^exact-vs-serial: (\S+)", body, re.M)
    r["match"] = "MATCH" if m and "MATCH" in m.group(1) else ("MISMATCH" if m else None)

    m = re.search(r"^accept rate: ([\d.]+)% \((\d+)/(\d+) drafted, (\d+) rounds\)", body, re.M)
    if m:
        r["accept_pct"] = float(m.group(1))
        r["accepted"] = int(m.group(2))
        r["drafted"] = int(m.group(3))
        r["rounds"] = int(m.group(4))

    m = re.search(r"^serial: ([\d.]+)s\s+mtp: ([\d.]+)s\s+speedup: ([\d.]+)x", body, re.M)
    if m:
        r["serial_s"] = float(m.group(1))
        r["mtp_s"] = float(m.group(2))
        r["speedup"] = float(m.group(3))
        if r.get("tokens"):
            r["serial_tok_s"] = r["tokens"] / r["serial_s"]
            r["mtp_tok_s"] = r["tokens"] / r["mtp_s"]

    m = re.search(r"^\[setup\] prompt tokens: (\d+)", body, re.M)
    r["prompt_tokens"] = int(m.group(1)) if m else None

    # explicit MTP prefill marker
    m = re.search(r"\[mtp\] prefill done\s+t=([\d.]+)s", body)
    r["mtp_prefill_s"] = float(m.group(1)) if m else None
    # serial first-token marker = prefill + 1 decode step
    m = re.search(r"\[serial\] 1/\d+ tok=\d+\s+t=([\d.]+)s", body)
    r["serial_first_tok_s"] = float(m.group(1)) if m else None
    # second serial token -> per-step delta
    m2 = re.search(r"\[serial\] 2/\d+ tok=\d+\s+t=([\d.]+)s", body)
    if m and m2:
        r["serial_step_s"] = float(m2.group(1)) - float(m.group(1))
        r["serial_prefill_est_s"] = float(m.group(1)) - r["serial_step_s"]

    m = re.search(r"WALL_EPOCH_DELTA=([\d.]+)", body)
    r["wall_s"] = float(m.group(1)) if m else None
    m = re.search(r"^real\s+([\d.]+)", body, re.M)
    if m:
        r["time_real_s"] = float(m.group(1))
    if r.get("wall_s") and r.get("serial_s") and r.get("mtp_s"):
        r["load_est_s"] = r["wall_s"] - r["serial_s"] - r["mtp_s"]

    r["eos"] = bool(re.search(r"\(eos\)|stop token \d+ committed", body))

    # per-draft-position acceptance from round lines (greedy prefix acceptance:
    # accepted==k  =>  positions 1..k accepted, position k+1 rejected)
    rounds = re.findall(r"\[mtp\] round \d+\s+primary=\d+\s+drafts=\[([^\]]*)\]\s+accepted (\d+)/(\d+)", body)
    per_pos = {}
    for drafts_s, acc, dep in rounds:
        n_drafts = len([x for x in drafts_s.split(",") if x.strip()])
        acc = int(acc)
        for pos in range(1, n_drafts + 1):
            d = per_pos.setdefault(pos, {"attempted": 0, "accepted": 0})
            d["attempted"] += 1
            if acc >= pos:
                d["accepted"] += 1
    r["per_position"] = {
        str(k): {"accepted": v["accepted"], "attempted": v["attempted"],
                 "pct": round(100.0 * v["accepted"] / v["attempted"], 1) if v["attempted"] else None}
        for k, v in sorted(per_pos.items())}
    r["round_lines_parsed"] = len(rounds)
    if r.get("tokens") and r.get("rounds"):
        r["tokens_per_target_forward"] = round(r["tokens"] / r["rounds"], 3)
    if r.get("accepted") and r.get("rounds"):
        r["avg_accepted_per_round"] = round(r["accepted"] / r["rounds"], 3)
    rows.append(r)

json.dump(rows, open("matrix_parsed.json", "w"), indent=2)

hdr = ("| cell | D | tok | MATCH | accept% | acc/draft | rounds | tok/fwd | "
       "serial t/s | mtp t/s | speedup | EOS |")
print(hdr)
print("|" + "---|" * 12)
for r in rows:
    print("| %s | %s | %s | %s | %s | %s/%s | %s | %s | %.1f | %.1f | %.2fx | %s |" % (
        r["cell"], r.get("depth"), r.get("tokens"), r.get("match"),
        r.get("accept_pct"), r.get("accepted"), r.get("drafted"), r.get("rounds"),
        r.get("tokens_per_target_forward"),
        r.get("serial_tok_s", 0), r.get("mtp_tok_s", 0), r.get("speedup", 0),
        "yes" if r.get("eos") else "budget"))

print()
print("PER-DRAFT-POSITION ACCEPTANCE")
for r in rows:
    pp = " ".join("pos%s %s%% (%s/%s)" % (k, v["pct"], v["accepted"], v["attempted"])
                  for k, v in r["per_position"].items())
    print("  %-12s %s" % (r["cell"], pp))

print()
print("PREFILL / LOAD")
for r in rows:
    print("  %-12s promptTok=%s mtp_prefill=%ss serial_first_tok=%ss serial_step=%ss "
          "serial_prefill_est=%ss wall=%ss load_est=%ss" % (
              r["cell"], r.get("prompt_tokens"), r.get("mtp_prefill_s"),
              r.get("serial_first_tok_s"),
              round(r["serial_step_s"], 4) if r.get("serial_step_s") else None,
              round(r["serial_prefill_est_s"], 4) if r.get("serial_prefill_est_s") else None,
              r.get("wall_s"), round(r["load_est_s"], 2) if r.get("load_est_s") else None))
