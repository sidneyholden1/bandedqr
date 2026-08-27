"""Tau-free boundary-bordering support for Dedalus, packaged as a single import.

Import this module *after* `import dedalus.public as d3` and it will:
  1. let Dedalus assemble the naturally-RECTANGULAR no-tau system (bulk equations +
     boundary/gauge rows, tall by the number of BCs) instead of demanding a square
     matrix, and fix the RHS-buffer sizing so the native timestepper handles it;
  2. register a least-squares matsolver 'taufree' that factors that tall system with
     a banded ride QR (RideQRRect) -- exploits the almost-banded structure directly,
     no normal-equations conditioning penalty.

With this in place a user writes the ordinary Dedalus problem but drops every tau
field, every Lift term, and the sphere `tau` bases -- keeping only genuine global
gauge scalars (e.g. the pressure nullspace tau_p) -- and passes matsolver='taufree'.

These three patches are what would live inside Dedalus behind a `tau_BC=False`
build flag; they are here as a self-contained shim so the run script stays clean.
"""
import os, sys, inspect, textwrap
import numpy as _np
import scipy.sparse as _sp
from scipy.linalg import solve_triangular as _solve_triangular
import dedalus.core.subsystems as _ss
import dedalus.core.system as _sys
from dedalus.tools.array import apply_sparse as _apply_sparse
from dedalus.libraries.matsolvers import SolverBase as _SolverBase, add_solver as _add_solver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bandqr import RideQRRect as _RideQRRect

# --- patch 1: allow rectangular (no-tau) assembly ---
_s1 = textwrap.dedent(inspect.getsource(_ss.Subproblem.build_matrices)).replace(
    "if valid_eqn.nnz != valid_var.nnz:", "if False:")
_g1 = _ss.__dict__.copy(); exec(_s1, _g1); _ss.Subproblem.build_matrices = _g1["build_matrices"]

# --- patch 2: RHS-side CoeffSystem buffers sized to rows (m), not cols (n) ---
_s2 = textwrap.dedent(inspect.getsource(_sys.CoeffSystem.__init__)).replace(
    "sp.LHS.shape[1]", "sp.LHS.shape[0]")
_g2 = _sys.__dict__.copy(); exec(_s2, _g2); _sys.CoeffSystem.__init__ = _g2["__init__"]

# --- patch 3: gather_inputs must use an n-sized (variable) buffer, not the m-sized one ---
def _gather_inputs(self, fields, out=None):
    views = self._input_field_views(tuple(fields))
    for buffer_view, field_view in views:
        _np.copyto(buffer_view, field_view)
    if out is None:
        n = self.pre_right_pinv.shape[0]
        out = _np.zeros((n, len(self.subsystems)), dtype=self.dtype)
    _apply_sparse(self.pre_right_pinv, self._input_buffer, axis=0, out=out)
    return out
_ss.Subproblem.gather_inputs = _gather_inputs


@_add_solver
class taufree(_SolverBase):
    """Least-squares matsolver for the tall tau-free operator: banded ride QR on the
    almost-banded system (bulk band + few dense BC rows).  Dense complete-QR fallback
    for any block the ride can't structure (e.g. the low-m near-dense corner)."""
    sparse = True
    def __init__(self, matrix, solver=None):
        A = _sp.csr_matrix(matrix)
        try:
            self.qr = _RideQRRect(A)
        except Exception:
            Ad = _np.asarray(A.todense()); self.n = Ad.shape[1]
            self.Q, self.R = _np.linalg.qr(Ad, mode='complete'); self.qr = None
    def solve(self, b):
        b = _np.asarray(b)
        if self.qr is not None:
            return self.qr.solve(b)
        return _solve_triangular(self.R[:self.n], (self.Q.conj().T @ b)[:self.n])
