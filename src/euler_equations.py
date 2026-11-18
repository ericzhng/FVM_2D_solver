import numpy as np
from numba import float64
from numba.experimental import jitclass

spec = [
    ("gamma", float64),
]


@jitclass(spec)
class EulerEquations:
    """
    Represents the 2D Euler equations for compressible fluid flow.

    This class provides the specific implementation for the Euler equations,
    including the conversion between conservative and primitive variables,
    flux calculation (Roe and HLLC), and wave speed estimation.

    Attributes:
        gamma (float): The ratio of specific heats (adiabatic index).
    """

    def __init__(self, gamma=1.4):
        """
        Initializes the EulerEquations object.

        Args:
            gamma (float, optional): The ratio of specific heats. Defaults to 1.4.
        """
        self.gamma = gamma

    def _cons_to_prim(self, U):
        """
        Converts a single conservative state vector to primitive variables.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].

        Returns:
            np.ndarray: Primitive state vector [rho, u, v, p].
        """
        rho, rho_u, rho_v, E = U
        # Floor density to prevent division by zero or negative pressure
        rho = max(rho, 1e-9)
        u = rho_u / rho
        v = rho_v / rho
        # Calculate pressure from total energy
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        return np.array([rho, u, v, p])

    def _prim_to_cons(self, P):
        """
        Converts a single primitive state vector to conservative variables.

        Args:
            P (np.ndarray): Primitive state vector [rho, u, v, p].

        Returns:
            np.ndarray: Conservative state vector [rho, rho*u, rho*v, E].
        """
        rho, u, v, p = P
        rho_u = rho * u
        rho_v = rho * v
        # Total energy E is the sum of internal and kinetic energy
        E = p / (self.gamma - 1) + 0.5 * rho * (u**2 + v**2)
        return np.array([rho, rho_u, rho_v, E])

    def _compute_flux(self, U, normal):
        """
        Calculates the physical flux across a face with a given normal.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The flux vector normal to the face.
        """
        rho, u, v, p = self._cons_to_prim(U)
        # Normal velocity
        vn = u * normal[0] + v * normal[1]
        # Total enthalpy H = (E + p) / rho
        H = (U[3] + p) / rho

        # Normal flux components
        F = np.array(
            [
                rho * vn,
                rho * vn * u + p * normal[0],
                rho * vn * v + p * normal[1],
                rho * vn * H,
            ]
        )
        return F

    def max_eigenvalue(self, U):
        """Return the maximum wave speed ``|v| + a`` of a cell (for the CFL step)."""
        rho = max(U[0], 1e-9)
        u = U[1] / rho
        v = U[2] / rho
        p = (self.gamma - 1) * (U[3] - 0.5 * rho * (u * u + v * v))
        a = np.sqrt(self.gamma * max(p, 1e-9) / rho)
        return np.sqrt(u * u + v * v) + a

    def _apply_wall_bc(self, U_inside, normal):
        """
        Applies a solid wall (reflective) boundary condition.

        This condition reflects the velocity normal to the wall while keeping the
        tangential velocity and thermodynamic properties (pressure, density) the same.

        Args:
            U_inside (np.ndarray): State vector of the interior cell.
            normal (np.ndarray): Normal vector of the boundary face.

        Returns:
            np.ndarray: The state vector of the ghost cell.
        """
        rho, u, v, p = self._cons_to_prim(U_inside)

        # Decompose velocity into normal and tangential components
        vn = u * normal[0] + v * normal[1]
        vt = u * -normal[1] + v * normal[0]

        # Reflect the normal velocity, keep tangential velocity
        vn_ghost = -vn
        vt_ghost = vt

        # Recompose the ghost velocity vector from the new normal and tangential components
        u_ghost = vn_ghost * normal[0] - vt_ghost * normal[1]
        v_ghost = vn_ghost * normal[1] + vt_ghost * normal[0]

        # Create the primitive state for the ghost cell
        P_ghost = np.array([rho, u_ghost, v_ghost, p])

        # Convert back to conservative variables
        return self._prim_to_cons(P_ghost)

    def apply_boundary_condition(self, U_inside, normal, bc_type, bc_value):
        # "inlet": 1
        # "outlet": 2
        # "wall": 3
        # "transmissive": 4
        # Ghost states are built from primitives and never mutate U_inside.
        if bc_type == 1:
            # Inlet: prescribed velocity, density and pressure from the interior.
            rho, u, v, p = self._cons_to_prim(U_inside)
            return self._prim_to_cons(np.array([rho, bc_value[0], bc_value[1], p]))
        elif bc_type == 2:
            # Outlet: prescribed static pressure, density and velocity from inside.
            rho, u, v, p = self._cons_to_prim(U_inside)
            return self._prim_to_cons(np.array([rho, u, v, bc_value[0]]))
        elif bc_type == 3:
            return self._apply_wall_bc(U_inside, normal)
        elif bc_type == 4:
            return U_inside
        else:
            raise ValueError("Invalid boundary condition type")

    # ------------------------------------------------------------------
    # Numerical fluxes. The ``*_into`` variants use scalar arithmetic only and
    # write into ``out`` (no allocations: they run inside parallel kernels).
    # ------------------------------------------------------------------
    def roe_flux(self, U_L, U_R, normal):
        """Roe flux (with Harten entropy fix); returns a new array."""
        out = np.empty(4)
        self.roe_flux_into(U_L, U_R, normal, out)
        return out

    def hllc_flux(self, U_L, U_R, normal):
        """HLLC flux; returns a new array."""
        out = np.empty(4)
        self.hllc_flux_into(U_L, U_R, normal, out)
        return out

    def roe_flux_into(self, U_L, U_R, normal, out):
        """Roe approximate Riemann flux across a face, written into ``out``.

        Args:
            U_L: Left conservative state ``[rho, rho u, rho v, E]``.
            U_R: Right conservative state.
            normal: Unit face normal (pointing from L to R).
            out: Array of length 4 receiving the flux.
        """
        gm1 = self.gamma - 1.0
        nx = normal[0]
        ny = normal[1]

        rL = max(U_L[0], 1e-9)
        uL = U_L[1] / rL
        vL = U_L[2] / rL
        pL = gm1 * (U_L[3] - 0.5 * rL * (uL * uL + vL * vL))
        HL = (U_L[3] + pL) / rL
        vnL = uL * nx + vL * ny
        vtL = -uL * ny + vL * nx

        rR = max(U_R[0], 1e-9)
        uR = U_R[1] / rR
        vR = U_R[2] / rR
        pR = gm1 * (U_R[3] - 0.5 * rR * (uR * uR + vR * vR))
        HR = (U_R[3] + pR) / rR
        vnR = uR * nx + vR * ny
        vtR = -uR * ny + vR * nx

        # Roe averages
        sL = np.sqrt(rL)
        sR = np.sqrt(rR)
        inv = 1.0 / (sL + sR)
        r = sL * sR
        u = (sL * uL + sR * uR) * inv
        v = (sL * vL + sR * vR) * inv
        H = (sL * HL + sR * HR) * inv
        q2 = u * u + v * v
        a = np.sqrt(max(gm1 * (H - 0.5 * q2), 1e-6))
        vn = u * nx + v * ny

        # Wave strengths
        dp = pR - pL
        dvn = vnR - vnL
        a2 = a * a
        w0 = (dp - r * a * dvn) / (2.0 * a2)
        w1 = r * (vtR - vtL)
        w2 = (rR - rL) - dp / a2
        w3 = (dp + r * a * dvn) / (2.0 * a2)

        # Wave speeds with Harten's entropy fix on the acoustic waves
        delta = 0.1 * a
        l0 = abs(vn - a)
        l3 = abs(vn + a)
        if l0 < delta:
            l0 = (l0 * l0 / delta + delta) / 2.0
        if l3 < delta:
            l3 = (l3 * l3 / delta + delta) / 2.0
        l1 = abs(vn)

        c0 = l0 * w0
        c1 = l1 * w1
        c2 = l1 * w2
        c3 = l3 * w3

        d0 = c0 + c2 + c3
        d1 = c0 * (u - a * nx) - c1 * ny + c2 * u + c3 * (u + a * nx)
        d2 = c0 * (v - a * ny) + c1 * nx + c2 * v + c3 * (v + a * ny)
        d3 = (
            c0 * (H - vn * a)
            - c1 * (u * ny - v * nx)
            + c2 * 0.5 * q2
            + c3 * (H + vn * a)
        )

        # Physical fluxes F(U_L) + F(U_R)
        mL = rL * vnL
        mR = rR * vnR
        out[0] = 0.5 * (mL + mR - d0)
        out[1] = 0.5 * (mL * uL + pL * nx + mR * uR + pR * nx - d1)
        out[2] = 0.5 * (mL * vL + pL * ny + mR * vR + pR * ny - d2)
        out[3] = 0.5 * (mL * HL + mR * HR - d3)

    def hllc_flux_into(self, U_L, U_R, normal, out):
        """HLLC flux across a face (Roe-averaged wave speeds), written into ``out``."""
        gm1 = self.gamma - 1.0
        nx = normal[0]
        ny = normal[1]

        rL = max(U_L[0], 1e-9)
        uL = U_L[1] / rL
        vL = U_L[2] / rL
        pL = gm1 * (U_L[3] - 0.5 * rL * (uL * uL + vL * vL))
        HL = (U_L[3] + pL) / rL
        vnL = uL * nx + vL * ny
        vtL = -uL * ny + vL * nx
        aL = np.sqrt(self.gamma * max(pL, 1e-9) / rL)

        rR = max(U_R[0], 1e-9)
        uR = U_R[1] / rR
        vR = U_R[2] / rR
        pR = gm1 * (U_R[3] - 0.5 * rR * (uR * uR + vR * vR))
        HR = (U_R[3] + pR) / rR
        vnR = uR * nx + vR * ny
        vtR = -uR * ny + vR * nx
        aR = np.sqrt(self.gamma * max(pR, 1e-9) / rR)

        # Roe-averaged speeds for the wave-speed estimates
        sL = np.sqrt(rL)
        sR = np.sqrt(rR)
        inv = 1.0 / (sL + sR)
        u_roe = (sL * uL + sR * uR) * inv
        v_roe = (sL * vL + sR * vR) * inv
        h_roe = (sL * HL + sR * HR) * inv
        vn_roe = u_roe * nx + v_roe * ny
        a_roe = np.sqrt(max(gm1 * (h_roe - 0.5 * (u_roe**2 + v_roe**2)), 1e-9))

        SL = min(vnL - aL, vn_roe - a_roe)
        SR = max(vnR + aR, vn_roe + a_roe)
        SM = (pR - pL + rL * vnL * (SL - vnL) - rR * vnR * (SR - vnR)) / (
            rL * (SL - vnL) - rR * (SR - vnR)
        )

        if SL >= 0.0:
            side = 0  # all waves move right: F(U_L)
        elif SM >= 0.0:
            side = 1  # star region, left of the contact
        elif SR > 0.0:
            side = 2  # star region, right of the contact
        else:
            side = 3  # all waves move left: F(U_R)

        if side == 0 or side == 3:
            if side == 0:
                r, u, v, p, H, vn = rL, uL, vL, pL, HL, vnL
            else:
                r, u, v, p, H, vn = rR, uR, vR, pR, HR, vnR
            m = r * vn
            out[0] = m
            out[1] = m * u + p * nx
            out[2] = m * v + p * ny
            out[3] = m * H
            return

        # Star region: F = F_K + S_K (U*_K - U_K)
        if side == 1:
            r, u, v, p, H, vn, vt, S = rL, uL, vL, pL, HL, vnL, vtL, SL
            rho_K, U1, U2, E = U_L[0], U_L[1], U_L[2], U_L[3]
        else:
            r, u, v, p, H, vn, vt, S = rR, uR, vR, pR, HR, vnR, vtR, SR
            rho_K, U1, U2, E = U_R[0], U_R[1], U_R[2], U_R[3]
        m = r * vn
        k = r * (S - vn) / (S - SM)
        Us1 = k * (SM * nx - vt * ny)
        Us2 = k * (SM * ny + vt * nx)
        Us3 = k * (E / r + (SM - vn) * (SM + p / (r * (S - vn))))
        out[0] = m + S * (k - rho_K)
        out[1] = m * u + p * nx + S * (Us1 - U1)
        out[2] = m * v + p * ny + S * (Us2 - U2)
        out[3] = m * H + S * (Us3 - E)
