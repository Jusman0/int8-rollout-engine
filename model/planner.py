"""
The host side of the planner (DESIGN.md §2 and §5), shared by every program
that drives the pendulum: model/golden.py today, and the co-simulation host
later. Because they all import this one file, they build the same candidate
plans from the same random numbers and quantize states the same way, so
their episodes can be compared number for number.

Every control tick:
  1. the host reads the real pendulum and quantizes its state to int8
     (quantize_state)
  2. the host builds N = 32 candidate plans of H int8 actions
     (Planner.candidates)
  3. an "engine" imagines all 32 futures and returns the index of the best
     one: the integer golden model, the float model, or the RTL
  4. the host applies that plan's first action and keeps the plan as the
     warm start for the next tick (Planner.commit)

run_episode() is that loop for one 200-step episode; the engine is passed in
as a function, so the same loop drives the float model, the integer model
and the hardware.
"""

import math

import numpy as np

N_CAND = 32          # candidate plans per tick (one per hardware lane)
SIGMA = 0.5          # warm-start noise, in units of u (u is in [-1, 1])
EPISODE_STEPS = 200  # Pendulum-v1 episode length

# int8 LSB sizes (DESIGN.md §2)
THETA_LSB = math.pi / 128    # rad
OMEGA_LSB = 8 / 127          # rad/s
U_LSB = 1 / 127              # action units; torque = 2u


def env_seed(seed, episode):
    """Seed for env.reset(): picks the episode's starting state."""
    return 7_000 + 100 * seed + episode


def rng_seed(seed, episode):
    """Seed for the host's random-number generator in this episode."""
    return 10_000 + 100 * seed + episode


def measured_state(obs):
    """Pendulum-v1 observes (cos theta, sin theta, omega). Returns the angle
    in [-pi, pi] and the angular velocity, as Python floats."""
    return math.atan2(float(obs[1]), float(obs[0])), float(obs[2])


def quantize_state(theta, omega):
    """Measured state -> int8 (DESIGN.md §2).

    theta: round to the nearest pi/128, then wrap into -128..127 (+pi and
           -pi are the same angle, so 128 becomes -128).
    omega: round to the nearest 8/127 rad/s, then saturate to -127..127.
    Python's round() sends an exact .5 to the even neighbour; a measured
    float practically never lands exactly on .5, and every host uses this
    same function, so they always agree.
    """
    theta_q = (round(theta / THETA_LSB) + 128) % 256 - 128
    omega_q = min(max(round(omega / OMEGA_LSB), -127), 127)
    return theta_q, omega_q


def torque(u_q):
    """int8 action -> the torque Pendulum-v1 expects, as a 1-element array."""
    return np.array([2 * u_q / 127], np.float32)


class Planner:
    """Builds the candidate plans for each tick and remembers the best one.

    plans are int64 arrays shaped (N_CAND, horizon): plans[n][t] is
    candidate n's action at imagined step t, in -127..127.
    """

    def __init__(self, rng, horizon, n_cand=N_CAND):
        self.rng = rng
        self.horizon = horizon
        self.n_cand = n_cand
        self.best_plan = None      # no warm start on the first tick
        self.plans = None

    def candidates(self):
        """The N_CAND plans for this tick (DESIGN.md §5 step 1)."""
        n, h = self.n_cand, self.horizon
        if self.best_plan is None:
            # first tick of the episode: nothing to build on, so uniform
            # random integers in [-127, 127] (the upper bound is exclusive)
            plans = self.rng.integers(-127, 128, size=(n, h))
        else:
            # candidate 0: last tick's best plan, shifted left one step
            # (that step has now been executed) with its last action repeated
            base = np.append(self.best_plan[1:], self.best_plan[-1])
            # candidates 1..31: candidate 0 plus Gaussian noise, clipped to
            # the legal range [-1, 1] and rounded to the nearest 1/127
            noise = SIGMA * self.rng.standard_normal((n - 1, h))
            noisy = np.clip(base / 127 + noise, -1.0, 1.0)
            plans = np.empty((n, h), np.int64)
            plans[0] = base
            plans[1:] = np.round(noisy * 127)
        self.plans = plans.astype(np.int64)
        return self.plans

    def commit(self, best_idx):
        """Keep the engine's chosen plan for the next tick's warm start and
        return the action to apply now (DESIGN.md §5 step 3)."""
        self.best_plan = self.plans[best_idx].copy()
        return int(self.best_plan[0])


def run_episode(env, engine, seed, episode, horizon):
    """One closed-loop episode; returns the total reward.

    engine(theta, omega, plans) -> best_idx gets the *measured* float state
    and does its own quantization if it needs int8, so the float model can
    use the unquantized state.
    """
    planner = Planner(np.random.default_rng(rng_seed(seed, episode)), horizon)
    obs, _ = env.reset(seed=env_seed(seed, episode))
    total = 0.0
    for _ in range(EPISODE_STEPS):
        theta, omega = measured_state(obs)
        best = engine(theta, omega, planner.candidates())
        u_q = planner.commit(best)
        obs, reward, _, _, _ = env.step(torque(u_q))
        total += float(reward)
    return total
