from __future__ import annotations

from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem


def main() -> None:
    # Synthetic scenario:
    # demand at slot 0 = 42, capacity per trip = 15, 3 shuttles depart at slot 1
    params = {
        "S": 15,
        "slot_resolution": 1,
        "Wmax_slots": 2,
        "p": 50.0,
        "lp_solver": "cplex_direct",
        "fill_first_epsilon": 0.0,
        "R_out": [42.0, 0.0, 0.0],
        "R_ret": [0.0, 0.0, 0.0],
    }
    candidate = {
        "yOUT[0,1]": 1.0,
        "yOUT[1,1]": 1.0,
        "yOUT[2,1]": 1.0,
        "yRET[0,1]": 0.0,
        "yRET[1,1]": 0.0,
        "yRET[2,1]": 0.0,
    }
    sp = ProblemSubproblem(params)
    res = sp.evaluate(candidate)
    diag = res.diagnostics or {}
    wait = float(diag.get("waiting_cost_slots", 0.0))
    penalty = float(diag.get("penalty_cost", 0.0))
    fill_eps = float(diag.get("fill_eps_cost", 0.0))
    obj = float(diag.get("objective_value", res.upper_bound or 0.0))
    assert wait >= -1e-9, f"waiting cost negative: {wait}"
    assert penalty >= -1e-9, f"penalty cost negative: {penalty}"
    assert (
        abs((wait + penalty + fill_eps) - obj) <= 1e-5
    ), "objective decomposition mismatch"
    print("Diagnostics OK.")


if __name__ == "__main__":
    main()
