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
