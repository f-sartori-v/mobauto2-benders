# Report figures

Every figure printed in the T.5.4 report is produced here, from a file that
ships in this repository. Nothing is drawn by hand, and no figure carries a
number that cannot be regenerated.

```bash
python scripts/report_figures/make_all.py          # writes outputs/figures/*.pdf and *.png
```

Only `matplotlib` is needed; the scripts parse the setup files themselves rather
than importing the model, so they run in a checkout with no solver installed.

| Script | Figure | Where its data comes from |
|---|---|---|
| `fig_demand_profile.py` | baseline demand profile of the instance family | `setups/base.yaml`, aggregated at the slot width the tactical model uses |
| `fig_demand_shapes.py` | the five generated shapes of the resolution study | `setups/generated/*.yaml`, written by `scripts/make_instances.py` |
| `fig_bound_interval.py` | the bound interval on the `Q = 3` minute-recourse instance | `data/measurements.json` |
| `fig_valuation_decomposition.py` | reporting error and decision error, in passenger-minutes | `data/measurements.json` |
| `fig_multiresolution_gain.py` | decision gain by shape, resolution and placement | `data/measurements.json` |
| `fig_penalty_window.py` | the `p_minutes` x `Wmax` frontier, and the zero-service cliff | `data/measurements.json` |

`data/measurements.json` holds the measured values and, for each block, the
decision-register entry that recorded it. Editing a value there changes the
figure and nothing else — the scripts hold no numbers of their own, so a figure
and the text beside it cannot drift apart silently.

The two figures driven by setup files are recomputed from the instances at every
run, so regenerating an instance regenerates its figure.
