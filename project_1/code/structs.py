import itertools
from dataclasses import dataclass, fields
from typing import NamedTuple, Protocol

from numba import typeof
from numba.typed import List


class ParamConstructor[P: NamedTuple](Protocol):
  def __call__(self, **kwargs) -> P: ...


class ParameterGridProtocol[P: NamedTuple](Protocol):
  param_type: type[P]

  def combos(self) -> list[P]: ...


@dataclass(frozen=True)
class ParameterGrid[P: NamedTuple]:
  param_type: ParamConstructor[P]

  def combos(self) -> list[P]:

    keys = [f.name for f in fields(self) if f.name != "param_type"]
    values = [getattr(self, k) for k in keys]

    return [
      self.param_type(**dict(zip(keys, combo))) for combo in itertools.product(*values)
    ]

  def combos_numba(self):
    combo_list = self.combos()

    combo_typedlist = List.empty_list(typeof(combo_list[0]))

    for combo in combo_list:
      combo_typedlist.append(combo)

    return combo_typedlist

class OptimizationRecord(NamedTuple):
  n: int
  alpha: float
  mean: float
  error: float
  bias: float

def records_to_markdown[P: NamedTuple](records: list[P], field_map: dict[str, tuple[str, str|None]]) -> str:

  fields = records[0]._fields

  headers = [field_map[f][0] for f in fields]
  md = "| " + " | ".join(headers) + " |\n"
  md += "|" + "|".join(["---"] * len(headers)) + "|\n"

  for r in records:
    row: list[str] = []
    for f in fields:
      value = getattr(r, f)
      fmt = field_map[f][1]

      if isinstance(value, float) and fmt is not None:
        row.append(format(value, fmt))
      else:
        row.append(str(value))
      
    md += "| " + " | ".join(row) + " |\n"
  
  return md