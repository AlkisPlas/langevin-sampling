from abc import ABC, abstractmethod
import numpy as np

class MALA(ABC):
    def __init__(self, d, eta=0.01, n_steps=100000, burn_in=1000, x0=None, seed=None):
        self.d = d
        self.eta = eta
        self.n_steps = n_steps
        self.burn_in = burn_in
        self.x0 = np.zeros(d) if x0 is None else np.array(x0)
        self.samples = np.zeros((n_steps, d))
        self.n_accepted = 0  # Track number of accepted proposals

        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def f(self, x):
        """Negative log-density (up to constant)."""
        pass

    @abstractmethod
    def grad_f(self, x):
        """Gradient of f."""
        pass

    # log target density up to a constant
    def log_pdf(self, x):
        return -self.f(x)

    # log multivariate Gaussian proposal density
    def log_q(self, x, x_prime):
        mu = x - self.eta * self.grad_f(x)
        diff = x_prime - mu
        return (
            -0.5 * self.d * np.log(2 * np.pi)
            -0.5 * self.d * np.log(2 * self.eta)
            -0.25 / self.eta * np.dot(diff, diff)
        )

    # proposal step
    def get_proposal(self, x):
        return (
            x - self.eta * self.grad_f(x) + np.sqrt(2 * self.eta) * np.random.randn(self.d)
        )

    # Metropolis-Hastings acceptance log ratio
    def compute_acceptance_log_ratio(self, x, x_prop):
        return (
            self.log_pdf(x_prop)
            + self.log_q(x_prop, x)
            - self.log_pdf(x)
            - self.log_q(x, x_prop)
        )

    def run(self):
        x = self.x0.copy()
        self.n_accepted = 0  # Reset counter

        for t in range(self.n_steps):
            x_prop = self.get_proposal(x)
            log_alpha = self.compute_acceptance_log_ratio(x, x_prop)

            if np.log(np.random.rand()) < log_alpha:
                x = x_prop
                self.n_accepted += 1

            self.samples[t] = x

        self.samples_post = self.samples[self.burn_in:]
        self.acceptance_rate = self.n_accepted / self.n_steps
        return self.samples_post
