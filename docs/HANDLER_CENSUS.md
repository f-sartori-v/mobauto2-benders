# Census of `except Exception` handlers

**231 handlers in `src/`.** Read-only survey, by AST rather than grep, classified by what
the `try` guards and what the handler does. Nothing edited yet — the point of the census
is that these are **not one problem**, and a mechanical narrowing pass would be a large
diff with real regression risk and almost no benefit.

Prior figures in `AUDIT_v4` were wrong: it said 165. The count was 225 at the round-1
merge `dbc01e2` and round 2 added 6.

## Distribution

| file | handlers |
|---|---|
| `benders/solver.py` | 93 |
| `problem/master_impl.py` | 73 |
| `problem/subproblem_impl.py` | 35 |
| `app.py` | 13 |
| `logging_config.py` | 7 |
| `config.py` | 5 |
| `benders/cplex_log.py` | 2 |
| `tolerances.py` | 2 |
| `manifest.py` | 1 |

By the shape of the guarded statement:

| shape | count |
|---|---|
| assignment with a numeric cast | 89 |
| assignment | 53 |
| `if` with a numeric cast | 23 |
| a call, not an assignment | 28 |
| `return <expr>` | 16 |
| `import` | 7 |
| `for` | 4 |

## Category A — wraps a required operation, fails silently *(3 handlers, act on these)*

These are the only ones with a correctness argument, and they share their failure shape
with `AUDIT_v4` C3: an operation that must succeed, whose failure produces a
plausible-looking wrong number rather than an error.

| site | what it guards | what a silent failure produces |
|---|---|---|
| `problem/subproblem_impl.py:1776` | `m.solutions.load_from(res)`, immediately before reading `m.dual.get(m.D_out[t], 0.0)` | the `0.0` default fires for every dual — an **all-zero slope vector**, i.e. a cut that constrains nothing, with no error |
| `problem/subproblem_impl.py:470` | the same load inside `solve_mw_dual`, before reading `dm_out` / `dm_ret` | wrong MW cut coefficients in the generator that was already the source of C3 |
| `problem/master_impl.py:754` | the same load after the master solve, before extracting stats and variable values | the model keeps the **previous iteration's** values; the loop reads a stale candidate and builds a cut from it |

**Not evidence that any of these fire.** They are latent. The fix is to make them loud —
raise, or log and mark the result invalid — not to assume they are firing today.

Contrast `problem/master_impl.py:703`, the same call, where failure triggers a retry with
solution loading enabled. That one is a genuine recovery and should stay.

## Category B — defensive around genuinely optional things *(keep)*

`logging_config.py` (6): tee writes and flushes to two streams, where a closed pipe is
expected. `app.py:20`: optional import of the problem implementations, which re-raises
with a useful message. Solver-attribute probes such as the loop over
`("logfile", "log_file", "_log_file", ...)` in `master_impl.py:716`, where absence is the
normal case across Pyomo backends.

## Category C — cast and lookup guards *(the bulk, ~180)*

`try: x = float(cfg.get(k)) / except Exception: x = default`, and variants. Individually
harmless, collectively the reason these files are hard to read. **36** are a single
statement with a silent handler — the exact shape of the existing `_cand_float` helper in
`subproblem_impl.py:1446`, which is precedent inside this codebase for collapsing them.

The win here is not narrowing `except Exception` to `except (TypeError, ValueError)` 180
times. It is removing the need for the handler by routing the access through a typed
accessor, which deletes lines instead of editing them.

## Category D — guards around code that cannot raise *(11)*

The `try` body makes no call at all. Candidates for deletion outright, but each needs
reading: a bare attribute access can raise `AttributeError` on a Pyomo component.

## Proposed order

1. **Category A, 3 handlers.** Correctness, tiny diff, and the D9 log-diff protocol (D27)
   can prove the result is unchanged. Do this first and alone.
2. **Category C via a helper**, file by file, starting with `master_impl.py`. Each step is
   provable by the baseline diff. This is where the line count actually falls.
3. **Category D**, opportunistically, when touching the surrounding code.
4. **Category B**: leave alone, and add a comment where the reason is not obvious.

Do not do a repo-wide narrowing pass. The audit's framing of this as one deferred chore is
what hid a real defect class inside a pile of `float()` guards.
