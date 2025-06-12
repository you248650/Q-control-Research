import numpy as np
import torch
from qutip import *


class DoubleHarmonicOscillator:
    def __init__(self, dim):
        """
        定数の定義((2)式と(3)式の間の各係数)
        """
        self.a = 0  # const
        self.b = 3  # const
        self.h = 5  # const
        self.gamma = 0.1  # const
        self.dim = dim  # matrix size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    """
    x operator's delta
    """

    def delta_x_1(self, m, n):
        if m == n + 1:
            return 1
        else:
            return 0

    def delta_x_2(self, m, n):
        if m == n - 1:
            return 1
        else:
            return 0

    """
    x operator matrix
    """

    def x(self):
        x_op = torch.zeros(
            (self.dim, self.dim), dtype=torch.complex128, device=self.device
        )

        for j in range(self.dim):
            for k in range(self.dim):
                x_op[k, j] = (1 / torch.sqrt(torch.tensor(2))) * (
                    (torch.sqrt(torch.tensor(k)) * self.delta_x_1(k, j))
                    + (torch.sqrt(torch.tensor(k + 1)) * self.delta_x_2(k, j))
                )

        return x_op

    def x2(self):
        x2_op = torch.matmul(self.x(), self.x())
        return x2_op

    def x3(self):
        x3_op = (
            torch.matmul(self.x2(), self.x()) + torch.matmul(self.x(), self.x2())
        ) / 2
        return x3_op

    def x4(self):
        x4_op = (
            torch.matmul(self.x3(), self.x()) + torch.matmul(self.x(), self.x3())
        ) / 2
        return x4_op

    """
    p operator's delta
    """

    def delta_p_1(self, m, n):
        if m == n + 1:
            return 1
        else:
            return 0

    def delta_p_2(self, m, n):
        if m == n - 1:
            return 1
        else:
            return 0

    """
    p operator matrix
    """

    def p(self):
        p_op = torch.zeros(
            (self.dim, self.dim), dtype=torch.complex128, device=self.device
        )

        for j in range(self.dim):
            for k in range(self.dim):
                p_op[k, j] = ((-1.0j) / torch.sqrt(torch.tensor(2))) * (
                    ((torch.sqrt(torch.tensor(k))) * self.delta_p_2(j, k))
                    - ((torch.sqrt(torch.tensor(k + 1))) * self.delta_p_1(j, k))
                )
        return p_op

    def p2(self):
        p2_op = torch.matmul(self.p(), self.p())
        return p2_op

    """
    System Hamiltonian
    """

    def System_Hamiltonian(self):
        phase = self.p2() / 2

        position_x4 = self.x4()
        position_x3 = 4 * self.a * self.x3()
        position_x2 = 2 * (3 * (self.a) ** (2) - (self.b) ** (2)) * self.x2()
        position_x = 4 * self.a * ((self.a) ** (2) - self.b ** (2)) * self.x()
        constant = ((self.a) ** (4)) - (2 * (self.a ** (2)) * self.b) + (self.b ** (4))

        position = (self.h / (self.b ** (4))) * (
            position_x4 - position_x3 + position_x2 - position_x + constant
        )

        system_hamiltonian = phase + position

        return system_hamiltonian

    """
    squueze operator's delta
    """

    def delta_squeeze_1(self, m, n):
        if m == n + 2:
            return 1
        else:
            return 0

    def delta_squeeze_2(self, m, n):
        if m == n - 2:
            return 1
        else:
            return 0

    """
    Squeezed Hamiltonian
    """

    def Squeezed_Hamiltonian(self):
        squ_mat = torch.zeros(
            (self.dim, self.dim), dtype=torch.complex128, device=self.device
        )

        for j in range(self.dim):
            for k in range(self.dim):
                squ_mat[k, j] = 1.0j * (
                    (
                        torch.sqrt(torch.tensor(k) * torch.tensor((k - 1)))
                        * self.delta_squeeze_2(j, k)
                    )
                    - (
                        torch.sqrt(torch.tensor((k + 1)) * torch.tensor((k + 2)))
                        * self.delta_squeeze_1(j, k)
                    )
                )

        return squ_mat

    """
    Stochastic Operator
    """

    def Stochastic_Operator(self):
        stochastic_operator = (
            torch.sqrt(torch.tensor(self.gamma, device=self.device)) * self.x2()
        )
        return stochastic_operator

    """
    Initial State
    """

    def Thermal_State(self):
        rho_init = thermal_dm(self.dim, 1)
        rho_init = np.array(rho_init)
        for i in range(self.dim):
            if i % 2 != 0:
                rho_init[i, i] = 0
        rho_init = torch.tensor(rho_init, device=self.device)
        rho_init = rho_init / torch.trace(rho_init)

        return rho_init

    # def Init_State(self):
    #     bits = int(torch.log2(torch.tensor(self.dim)))
    #     rho_init = torch.eye(1, dtype=torch.complex128, device=self.device)

    #     for _ in range(bits):
    #         random_state = torch.rand(2, 2) + 1j * torch.rand(2, 2)
    #         random_state = random_state.to(self.device)
    #         rho_init = torch.kron(rho_init, random_state)

    #     rho_init = self.positive_semidefinite(rho_init)
    #     rho_init = (rho_init + torch.conj(rho_init.T)) / 2
    #     rho_init = rho_init / torch.trace(rho_init)

    #     return rho_init

    # def positive_semidefinite(self, rho_init):
    #     rho_init = (rho_init + torch.conj(rho_init).T) / 2

    #     e, v = torch.linalg.eigh(rho_init)

    #     e = torch.clamp(e, min=1e-5).to(torch.complex128)

    #     rho_init = torch.matmul(v, torch.matmul(e.diag_embed(), torch.conj(v).T))

    #     return rho_init
