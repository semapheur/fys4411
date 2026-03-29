import json
from pathlib import Path
from typing import Literal


def upsert_params(
  path: str, model: Literal["harmonic", "repulsive"], new_params: dict[int, float]
):
  path_ = Path(path)
  if not path_.suffix == ".json":
    raise ValueError(f"path must be a json file, got {path}")

  if Path(path).exists():
    try:
      with open(path, "r") as f:
        data = json.load(f)

    except json.JSONDecodeError:
      data = {}
  else:
    data = {}

  model_dict = data.setdefault(model, {})
  model_dict.update(new_params)

  tmp_path = path_.with_suffix(".tmp")
  with open(tmp_path, "w") as f:
    json.dump(data, f)

  tmp_path.replace(path_)
