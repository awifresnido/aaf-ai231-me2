import json, sys
sys.path.insert(0, "src")
from src.vcm_data_loader import create_label_mapping

mapping = create_label_mapping("configs/ontology.json", "leaf")
# index -> key (model output order)
labels = [k for k, v in sorted(mapping.items(), key=lambda kv: kv[1])]
assert len(labels) == 33, f"expected 33, got {len(labels)}"

out = {"label_mode": "leaf", "num_classes": len(labels), "labels": labels}
json.dump(out, open("exports/E1_s0/labels.json", "w"), indent=2)

# also embed into deploy.json
d = json.load(open("exports/E1_s0/deploy.json"))
d["labels"] = labels
json.dump(d, open("exports/E1_s0/deploy.json", "w"), indent=2)

print("num_classes:", len(labels))
print("labels:", labels)
