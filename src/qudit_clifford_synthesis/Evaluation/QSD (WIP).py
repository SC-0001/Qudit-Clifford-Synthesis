"""
TODO: Implement.
Planning to implement the Quantum Shannon Decomposition (QSD) to decompose arbitrary multi-qudit gates into a sequence of parameterized
one and two-qudit gates. (paulis)

# Work-in-Progress Notes ------------------------------------------------------------------------------------------------------------------------
1. 
    Via Regtangular CSD, decompose some U ∈ U(d^N) to an alternating sequence of #1 direct-sums of unitary blocks 
    and #2 cosine-sine matrices. Recursively apply rectangular CSD onto the type #1 matrices until each type #1
    matrix in the sequence is a direct sum of U(d^{N-1}) matrices. In the end, we should have Σ_{n=1}^d 2^{n-1} 
    matrices of type #2 and 2^d matrices of type #1.
        [!] 
            Apparently there exists a generalized CSD that doesn't require recursion.
            I'm yet to find a package that performs this "generalized CSD". Consider implementing it.
2. 
    Each matrix of type #1 is a QMUX—depending on the state of the 1st qudit, different U(d^{N-1}) unitaries applied
    to the rest of the qudits. If the implementation of any U(d^{N-1}) unitary is non-trivial, we must recursively apply
    this entire CSD+QMUX procedure again.
3. 
    Each matrix of type #2 is a QMUX too, but N-1 qudits each applying a conditional R_y^{0,d-1} gate onto the 1st qudit.

Ref:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cossin.html
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.012325
https://arxiv.org/abs/quant-ph/0406176

#------------------------------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
from scipy.linalg import cossin
from scipy.stats import unitary_group

import qudit_clifford_synthesis.Essentials.QuditCirc