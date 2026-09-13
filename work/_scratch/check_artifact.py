import json

with open("data/experiments/EXP-002_bbq_stereotype_direction/output/operation_vector.json") as f:
    d = json.load(f)

print("top-level keys:", list(d.keys()))
print("directions:", list(d["h1_evaluation"]["directions"].keys()))
print("baseline_margins:", d["h1_evaluation"]["baseline_margins"])
print("n_limitations:", len(d["limitations"]))
print("fused_working_set:", d["fused_working_set"])
