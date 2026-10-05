"""Probe which element keywords this PlantUML build accepts."""
import subprocess
import tempfile
from pathlib import Path

JAR = Path(__file__).resolve().parent / "plantuml.jar"

CASES = {
    "rectangle": '@startuml\nrectangle "A" as A\nA --> B\n@enduml',
    "node": '@startuml\nnode "A" as A\nA --> B\n@enduml',
    "folder": '@startuml\nfolder "A" as A\nA --> B\n@enduml',
    "database": '@startuml\ndatabase "A" as A\nA --> B\n@enduml',
    "file": '@startuml\nfile "A" as A\nA --> B\n@enduml',
    "component": '@startuml\ncomponent "A" as A\nA --> B\n@enduml',
    "artifact": '@startuml\nartifact "A" as A\nA --> B\n@enduml',
    "system": '@startuml\nsystem "A" as A\nA --> B\n@enduml',
    "system noalias": '@startuml\nsystem "A"\nA --> B\n@enduml',
    "C4 with teoz pragma": '@startuml\n!pragma teoz\nsystem "A" as A\ndatabase "B" as B\nA --> B\n@enduml',
    "C4 library include": '@startuml\n!include <C4/C4_Context>\nSystem_Boundary(b) {\n  System(sys, "Sys")\n}\nPerson(u, "User")\nRel(u, sys, "uses")\n@enduml',
    "card": '@startuml\ncard "A" as A\nA --> B\n@enduml',
    "class keyword": '@startuml\nclass "A" as A\nA --> B\n@enduml',
    "usecase": '@startuml\nusecase "A" as A\nA --> B\n@enduml',
}


def run(src):
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.puml"
        f.write_text(src + "\n", encoding="utf-8")
        p = subprocess.run(
            ["java", "-jar", str(JAR), "-checkonly", "-charset", "UTF-8", str(f)],
            capture_output=True, text=True, timeout=90,
        )
        blob = ((p.stdout or "") + (p.stderr or "")).lower()
        return "contains errors" not in blob


print(f"{'keyword':22s} result")
print("-" * 46)
for name, src in CASES.items():
    try:
        ok = run(src)
    except subprocess.TimeoutExpired:
        ok = None
    v = "OK" if ok else ("TIMEOUT" if ok is None else "** REJECTED **")
    print(f"{name:22s} {v}")