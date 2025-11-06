def test_imports():
    import mobauto2_benders as m
    from mobauto2_benders.benders.solver import BendersSolver
    from mobauto2_benders.benders.master import MasterProblem
    from mobauto2_benders.benders.subproblem import Subproblem
    from mobauto2_benders.config import load_config

    assert hasattr(m, "__version__")
    assert callable(load_config)
    # Abstract base classes import
    assert MasterProblem
    assert Subproblem
    assert BendersSolver

