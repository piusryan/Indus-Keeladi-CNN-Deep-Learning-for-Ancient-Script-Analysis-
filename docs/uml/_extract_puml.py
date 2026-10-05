"""Split PlantUML blocks out of docs/UML_DIAGRAMS.md into docs/uml/*.puml files."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DOC = ROOT / "docs" / "UML_DIAGRAMS.md"
OUT = ROOT / "docs" / "uml"
OUT.mkdir(parents=True, exist_ok=True)

text = DOC.read_text(encoding="utf-8")
blocks = re.findall(r"```plantuml\n(.*?)```", text, re.S)
print(f"found {len(blocks)} plantuml blocks")

titles = [
    "01_system_context", "02_component", "03_class_preprocessing_model",
    "04_class_training_evaluation", "05_class_audit_matching",
    "06_sequence_pipeline", "07_sequence_training", "08_sequence_evaluation",
    "09_activity_preprocessing", "10_activity_audit",
    "11_state_model_lifecycle", "12_deployment", "13_packages", "14_use_case",
]

for i, block in enumerate(blocks):
    name = titles[i] if i < len(titles) else f"diagram_{i + 1:02d}"
    path = OUT / f"{name}.puml"
    path.write_text(block.strip() + "\n", encoding="utf-8")
    print(f"  wrote {path.name}  ({len(block.splitlines())} lines)")

if len(blocks) != len(titles):
    print(f"WARNING: expected {len(titles)} diagrams, found {len(blocks)}")