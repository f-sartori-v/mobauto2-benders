# mobauto2-benders
Bender decomposition approach of MobAuto2 project

Quick start
- CLI: `python -m mobauto2_benders run` reads `configs/default.yaml` by default.
- Python: use the convenience runner that loads the default YAML and respects all options (e.g., Magnanti–Wong) without passing flags:

```
from mobauto2_benders import run

result = run()  # uses configs/default.yaml
print(result.status, result.iterations, result.best_lower_bound, result.best_upper_bound)
```

To customize, edit `configs/default.yaml` (e.g., set `subproblem.params.use_magnanti_wong: true`).
