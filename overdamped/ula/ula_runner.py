import numpy as np
from abc import ABC, abstractmethod

class ULA(ABC):
    def __init__(self, d, eta=0.01, n_steps=100000, burn_in=1000, x0=None, seed=None):
        """
        Base class for Unadjusted Langevin Algorithm (ULA)

        Parameters:
        -----------
        d : int
            Dimension of the target distribution
        eta : float
            Step size
        n_steps : int
            Total number of steps
        burn_in : int
            Number of initial steps to discard
        x0 : np.array or None
            Initial point (default zeros)
        seed : int or None
            Random seed for reproducibility
        """
        self.d = d
        self.eta = eta
        self.n_steps = n_steps
        self.burn_in = burn_in
        self.x0 = np.zeros(d) if x0 is None else np.array(x0)
        self.samples = np.zeros((n_steps, d))
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def grad_f(self, x):
        """
        Gradient of the negative log-target (potential function).
        Must be implemented in subclass.
        """
        pass

    def run(self):
        """
        Run the Unadjusted Langevin Algorithm and store all samples.
        """
        x = self.x0.copy()
        for t in range(self.n_steps):
            x = x - self.eta * self.grad_f(x) + np.sqrt(2 * self.eta) * np.random.randn(self.d)
            self.samples[t] = x

        # Post-burn-in samples
        self.samples_post = self.samples[self.burn_in:]
        return self.samples_post
