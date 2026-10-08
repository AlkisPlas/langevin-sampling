from abc import ABC, abstractmethod
import numpy as np

class BAOAB(ABC):
    def __init__(self, d, eta, gamma, n_steps=100000, burn_in=1000, 
                 x0=None, v0 = None, seed=None):
        self.d = d
        self.eta = eta
        self.gamma = gamma # friction
        self.n_steps = n_steps
        self.burn_in = burn_in
        self.x0 = np.zeros(d) if x0 is None else np.array(x0)
        self.v0 = np.zeros(d) if v0 is None else np.array(v0)
        self.samples = np.zeros((n_steps, d))
        # Precompute Ornstein–Uhlenbeck coefficients
        self.exp_gamma = np.exp(-gamma * eta)
        self.sigma = np.sqrt(1 - self.exp_gamma**2)

        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def f(self, x):
        pass

    @abstractmethod
    def grad_f(self, x):
        pass

    # Update velocities based on potential forces
    def half_force(self, x):
        return 0.5 * self.eta * self.grad_f(x)
    
    # Update positions based on velocities
    def half_drift(self, v):
        return 0.5 * self.eta * v
    
    # Update velocities due to friction and random noise 
    def ornstein_uhlenbeck(self, v):
        return self.exp_gamma * v + self.sigma * np.random.randn(self.d)

    def run(self):
        x = self.x0.copy()
        v = self.v0.copy()
        # The force of the last B step is reused in the first B step of the next one,
        # so each step needs one new gradient.
        force = self.half_force(x)
        for t in range(self.n_steps):
            # B: half force
            v -= force
            # A: half drift
            x += self.half_drift(v)
            # O: Ornstein–Uhlenbeck
            v = self.ornstein_uhlenbeck(v)
            # A: half drift
            x += self.half_drift(v)
            # B: half force
            force = self.half_force(x)
            v -= force

            self.samples[t] = x

        # Post-burn-in samples
        self.samples_post = self.samples[self.burn_in:]
        return self.samples_post