# [Written by Claude]
"""Worker processes of a multi-processing sweep must use the solver chosen in the main process.

Each worker builds its own SolveRoutine. Before this test existed, a solver chosen with set_solver()
was not passed on (the workers used the automatic choice), and SolverSparta.duplicate() dropped the
CPU/GPU mode and the symmetry setting."""
import pickle

import pytest

from emerge._emerge import solver as S
from emerge._emerge.physics.microwave import microwave_3d as mw3d

pytestmark = pytest.mark.skipif(not S._SPARTA_AVAILABLE, reason="SPARTA is not installed")


def _sparta_routine():
    routine = S.AutomaticRoutine()
    routine.set_solver(S.EMSolver.SPARTA)
    solver = routine.forced_solver[0]
    solver.set_options("CPU")
    solver.set_symmetry(False)
    return routine


def test_sparta_duplicate_keeps_settings():
    solver = _sparta_routine().forced_solver[0]
    solver.settings["refinement_steps"] = 2
    copy = solver.duplicate()
    assert copy.mode == "CPU"
    assert copy._csym is False
    assert copy.settings["refinement_steps"] == 2
    assert copy.settings is not solver.settings


def test_worker_routine_uses_forced_solver():
    parent = _sparta_routine()
    parent.force_symmetric = True
    choice = pickle.loads(pickle.dumps(mw3d._solver_choice(parent)))   # as sent to a spawned worker

    worker = S.AutomaticRoutine()._configure_routine("MP", proc_nr=2)
    mw3d._apply_worker_choice(worker, choice)

    assert len(worker.forced_solver) == 1
    forced = worker.forced_solver[0]
    assert isinstance(forced, S.SolverSparta)
    assert forced.mode == "CPU" and forced._csym is False
    assert forced is not parent.forced_solver[0]
    assert worker.solvers[S.EMSolver.SPARTA] is forced     # reset() reaches the forced solver
    assert worker.force_symmetric is True
    assert forced.pre == worker.pre


def test_no_choice_keeps_automatic_routine():
    worker = S.AutomaticRoutine()._configure_routine("MP", proc_nr=1)
    before = list(worker.forced_solver)
    mw3d._apply_worker_choice(worker, None)
    assert worker.forced_solver == before
