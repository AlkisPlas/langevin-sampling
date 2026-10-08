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

    # log multivariate Gaussian proposal density, given the gradient at x
    def log_q(self, x, x_prime, grad_x):
        mu = x - self.eta * grad_x
        diff = x_prime - mu
        return (
            -0.5 * self.d * np.log(2 * np.pi)
            -0.5 * self.d * np.log(2 * self.eta)
            -0.25 / self.eta * np.dot(diff, diff)
        )

    # proposal step, given the gradient at x
    def get_proposal(self, x, grad_x):
        return (
            x - self.eta * grad_x + np.sqrt(2 * self.eta) * np.random.randn(self.d)
        )

    # Metropolis-Hastings acceptance log ratio
    def compute_acceptance_log_ratio(self, x, x_prop, log_pdf_x, log_pdf_prop,
                                     grad_x, grad_prop):
        return (
            log_pdf_prop
            + self.log_q(x_prop, x, grad_prop)
            - log_pdf_x
            - self.log_q(x, x_prop, grad_x)
        )

    def run(self):
        x = self.x0.copy()
        self.n_accepted = 0  # Reset counter

        # log density and gradient of the current state are kept between steps,
        # so each proposal needs one new gradient and one new log density.
        log_pdf_x = self.log_pdf(x)
        grad_x = self.grad_f(x)
        for t in range(self.n_steps):
            x_prop = self.get_proposal(x, grad_x)
            log_pdf_prop = self.log_pdf(x_prop)
            grad_prop = self.grad_f(x_prop)
            log_alpha = self.compute_acceptance_log_ratio(
                x, x_prop, log_pdf_x, log_pdf_prop, grad_x, grad_prop)

            if np.log(np.random.rand()) < log_alpha:
                x = x_prop
                log_pdf_x = log_pdf_prop
                grad_x = grad_prop
                self.n_accepted += 1

            self.samples[t] = x

        self.samples_post = self.samples[self.burn_in:]
        self.acceptance_rate = self.n_accepted / self.n_steps
        return self.samples_post
