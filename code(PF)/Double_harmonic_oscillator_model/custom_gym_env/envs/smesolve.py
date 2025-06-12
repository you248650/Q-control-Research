import numpy as np
import torch

from custom_gym_env.envs.double_harmonic_oscillator import *


class Simulator:
    def __init__(self, process_id, dim):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hyperparameters
        self.time = torch.tensor(0.001, device=self.device)
        self.step = torch.tensor(100, device=self.device)
        self.dt = self.time / self.step
        self.dim = torch.tensor(dim, device=self.device)
        self.gamma = torch.tensor(0.1, device=self.device)

        self.PF_nums = torch.tensor(40, device=self.device)
        self.sigma = torch.tensor(0.7, device=self.device)

        self.process_id = process_id

        # DoubleHarmonicOscillator class to instance
        self.oscillator = DoubleHarmonicOscillator(dim)

        # System Hamiltonian
        self.system_hamiltonian = self.oscillator.System_Hamiltonian()

        # Squeezed Hamiltonian
        self.squeezed_hamiltonian = self.oscillator.Squeezed_Hamiltonian()

        # Stochastic Operator
        self.A = self.oscillator.Stochastic_Operator()

    # Super operators
    def dag(self, A, mode):
        if mode == "SME simulator":
            i = 0
        elif mode == "Particle Filter":
            i = 1
        return torch.conj(A.transpose(i, i + 1))

    def commutator(self, H, R):
        return H @ R - R @ H

    def anticommutator(self, A, R, mode):
        return A @ R + R @ self.dag(A, mode)

    def Hop(self, A, R, mode):  # (4)式
        return (
            self.anticommutator(A, R, mode)
            - torch.diagonal(self.anticommutator(A, R, mode), dim1=-2, dim2=-1)
            .sum(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            * R
        )

    def Dop(self, A, R, mode):  # (5)式
        return A @ R @ self.dag(A, mode) - 0.5 * self.anticommutator(
            R, self.dag(A, mode) @ A, mode
        )

    # Simulator for stochastic master equation
    def SMEsolve(self, rho_init, action, sim_step=None, dW=None, mode="SME simulator"):
        action = torch.tensor(action, dtype=torch.float64, device=self.device)

        # Hamiltonian
        H = self.system_hamiltonian + action * self.squeezed_hamiltonian

        if isinstance(rho_init, np.ndarray):
            rho_init = torch.tensor(
                rho_init, dtype=torch.complex128, device=self.device
            )

        if mode == "SME simulator":
            # Append init state
            d_rho = torch.zeros(
                (sim_step + 1, self.dim, self.dim),
                dtype=torch.complex128,
                device=self.device,
            )

            d_rho[0] = rho_init.clone()

            for t in range(sim_step):
                self.dW = torch.normal(0.0, self.dt, generator=None).to(self.device)
                # Solve SME
                d_rho[t + 1] = (
                    -1j * self.commutator(H, d_rho[t]) * self.dt
                    + self.Dop(self.A, d_rho[t], mode) * self.dt
                    + self.Hop(self.A, d_rho[t], mode) * self.dW
                )

            rho = rho_init + d_rho[-1]

            # Calculate measurement current
            dQ = torch.real(
                (torch.trace(self.A @ rho)) * self.dt
                + (1 / torch.sqrt(4 * self.gamma)) * self.dW
            )

            return rho, dQ

        elif mode == "Particle Filter":
            d_PF_rho = torch.zeros(
                (sim_step + 1, sim_step, self.dim, self.dim),
                dtype=torch.complex128,
                device=self.device,
            )

            d_PF_rho[0] = rho_init.clone()

            H = torch.stack([H] * sim_step, dim=0)
            A = torch.stack([self.A] * sim_step, dim=0)

            for t in range(sim_step):
                # Solve SME
                d_PF_rho[t + 1] = (
                    -1j * self.commutator(H, d_PF_rho[t]) * self.dt
                    + self.Dop(A, d_PF_rho[t], mode) * self.dt
                    + self.Hop(A, d_PF_rho[t], mode) * dW[t]
                )

            PF_rho = rho_init + d_PF_rho[-1]

            return PF_rho

        elif mode == "Measurement current for est_rho":
            # Calculate measurement current
            PF_dQ = torch.real(
                (torch.trace(self.A @ rho_init)) * self.dt
                + (1 / torch.sqrt(4 * self.gamma)) * self.dW
            )

            return PF_dQ

    def Particle_Filter(self, PF_rho, action, dQ):
        r = torch.rand(1)
        if r < 0.03:
            self.PF_rho = torch.stack(
                [self.oscillator.Init_State() for _ in range(self.PF_nums)]
            )

        PF_measure_operator = torch.stack(
            [torch.trace(self.A @ p_r).real for p_r in PF_rho]
        )
        PF_dQ = torch.tensor(
            [PF_measure * self.dt for PF_measure in PF_measure_operator],
            device=self.device,
        )

        dQ_dif = torch.sqrt(4 * self.gamma) * (dQ - PF_dQ)

        r = torch.randint(0, len(PF_rho), (len(PF_rho),))

        PF_rho = self.SMEsolve(
            PF_rho, action, len(PF_rho), dQ_dif[r], mode="Particle Filter"
        )

        PF_measure_operator = torch.stack(
            [torch.trace(self.A @ p_r).real for p_r in PF_rho]
        )

        PF_dQ = torch.tensor(
            [
                PF_measure_operator[j] * self.dt
                + (1 / torch.sqrt(4 * self.gamma)) * dQ_dif[r[j]]
                for j in range(len(PF_measure_operator))
            ],
            device=self.device,
        )

        weight = self.update_weight(PF_dQ, dQ, self.sigma)
        weight = weight / torch.sum(weight)

        indexes = self.systematic_resample(weight)
        PF_rho = self.resample_from_index(PF_rho, indexes)

        weighted_sum = torch.sum(PF_rho * weight.unsqueeze(-1).unsqueeze(-1), dim=0)
        sum_of_weights = torch.sum(weight)
        est_rho = weighted_sum / sum_of_weights

        return est_rho, PF_rho

    def update_weight(self, particles, dQ, sigma):
        weights = torch.sqrt(torch.tensor(2 * torch.pi) * sigma**2) ** (-1) * torch.exp(
            -(dQ - particles) / (2 * sigma**2)
        )

        for i in range(len(weights)):
            if torch.isnan(weights[i]):
                weights[i] = 0

        weights += 1e-300
        weights = weights / torch.sum(weights)

        return weights

    def systematic_resample(self, weights):
        N = len(weights)

        positions = (torch.arange(N).float() + torch.rand(N)) / N

        indexes = torch.zeros(N, dtype=torch.int64)
        cumulative_sum = torch.cumsum(weights, dim=0)

        i, j = 0, 0
        while i < N and j < len(cumulative_sum):
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        return indexes

    def resample_from_index(self, particles, indexes):
        particles[:] = particles[indexes]

        return particles
