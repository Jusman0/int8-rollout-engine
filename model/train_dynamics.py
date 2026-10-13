"""
Train the learned pendulum dynamics model (DESIGN.md §3): a 3 -> 16 -> 2 MLP
in plain NumPy, float32.

  input   (theta, omega, u)     angle [rad], angular velocity [rad/s],
                                action u in [-1, 1] (torque = 2u)
  output  (d_theta, d_omega)    how much the state changes in one 0.05 s step

The planner uses this model to *imagine* where the pendulum goes under a
candidate plan, without touching the real pendulum.

Data: 30,000 transitions of gymnasium Pendulum-v1 with uniformly random
actions. The pendulum is reset to a random state every 25 steps, so the data
covers the whole circle and not just one long swing. d_theta is wrapped to
[-pi, pi), so a step across the +pi / -pi seam counts as a small change and
not as a jump of almost 2*pi.

Training: inputs and targets are standardized (subtract the mean, divide by
the standard deviation) so all of them are about the same size, then plain
mean-squared error with the Adam optimizer, 60 epochs, batches of 256. Every
random number comes from one seeded generator, so a rerun on the same
machine gives the same weights.

Writes model/dynamics_fp32.npz:
  W1 (16, 3), b1 (16,)   layer 1; W1[j][i] is the weight from input i to
                         hidden neuron j (the same indexing as DESIGN.md §4)
  W2 (2, 16), b2 (2,)    layer 2; W2[k][j] is from hidden j to output k
  xm, xs (3,)            input mean and standard deviation
  ym, ys (2,)            target mean and standard deviation
  a1_max ()              largest layer-1 pre-activation over the training
                         data; golden.py sets the hidden scale s_h from it

Run from the repo root:  python3 model/train_dynamics.py
"""

from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SEED = 0
N_TRANSITIONS = 30_000
RESET_EVERY = 25
SIZES = [3, 16, 2]          # inputs, hidden neurons, outputs
EPOCHS = 60
BATCH = 256
LR = 1e-3

# Adam's usual constants
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def wrap_angle(a):
    """Map any angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def angle_of(obs):
    """Pendulum-v1 observes (cos theta, sin theta, omega); recover theta."""
    return np.arctan2(obs[1], obs[0])


def collect(rng, first_seed, n):
    """n random-action transitions. Returns X = (theta, omega, u), Y = change."""
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=first_seed)
    states, actions, next_states = [], [], []
    for i in range(n):
        u = rng.uniform(-1, 1)
        nxt, _, _, _, _ = env.step(np.array([2 * u], np.float32))
        states.append([angle_of(obs), obs[2]])
        actions.append(u)
        next_states.append([angle_of(nxt), nxt[2]])
        obs = nxt
        if (i + 1) % RESET_EVERY == 0:
            # the reset seed comes from rng too, so the whole data set
            # depends only on SEED
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    s = np.array(states, np.float32)
    a = np.array(actions, np.float32)[:, None]
    s2 = np.array(next_states, np.float32)
    x = np.hstack([s, a])
    y = s2 - s
    y[:, 0] = wrap_angle(y[:, 0])
    return x, y


def init_layers(rng):
    """He initialization. During training a layer is (W, b) with W shaped
    (inputs, outputs), so a batch goes through it as x @ W + b.

    np.sqrt() returns a NumPy float64 scalar, and multiplying a float32 array
    by one silently gives a float64 array. Wrapping it in np.float32() keeps
    the whole network (and all of training) in float32."""
    return [(rng.standard_normal((i, o)).astype(np.float32) * np.float32(np.sqrt(2 / i)),
             np.zeros(o, np.float32))
            for i, o in zip(SIZES, SIZES[1:])]


def forward(layers, x):
    """Returns the input and every layer's output (ReLU on all but the last)."""
    outs = [x]
    for k, (w, b) in enumerate(layers):
        x = x @ w + b
        if k < len(layers) - 1:
            x = np.maximum(x, 0)
        outs.append(x)
    return outs


def train(layers, x, y, rng):
    """Mean-squared-error regression with Adam (first and second moment
    estimates m, v, with the usual bias correction)."""
    m = [[np.zeros_like(w), np.zeros_like(b)] for w, b in layers]
    v = [[np.zeros_like(w), np.zeros_like(b)] for w, b in layers]
    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(len(x))
        for start in range(0, len(x), BATCH):
            batch = order[start:start + BATCH]
            outs = forward(layers, x[batch])
            # gradient of the mean squared error with respect to the output
            g = 2 * (outs[-1] - y[batch]) / len(batch)
            t += 1
            # backpropagation, last layer first
            for j in range(len(layers) - 1, -1, -1):
                w, b = layers[j]
                grads = [outs[j].T @ g, g.sum(0)]
                if j > 0:
                    # pass the gradient back through W (before it is
                    # updated) and through the ReLU
                    g = (g @ w.T) * (outs[j] > 0)
                params = [w, b]
                for p in range(2):
                    m[j][p] = BETA1 * m[j][p] + (1 - BETA1) * grads[p]
                    v[j][p] = BETA2 * v[j][p] + (1 - BETA2) * grads[p] ** 2
                    m_hat = m[j][p] / (1 - BETA1 ** t)
                    v_hat = v[j][p] / (1 - BETA2 ** t)
                    params[p] = params[p] - LR * m_hat / (np.sqrt(v_hat) + EPS)
                layers[j] = (params[0], params[1])
        if (epoch + 1) % 10 == 0:
            mse = float(((forward(layers, x)[-1] - y) ** 2).mean())
            print(f"epoch {epoch + 1:3d}: training MSE (standardized units) {mse:.5f}")
    return layers


def rmse_report(name, layers, x, y, xm, xs, ym, ys):
    """Root-mean-square one-step error, in physical units and in int8 LSBs."""
    pred = forward(layers, (x - xm) / xs)[-1] * ys + ym
    err = pred - y
    err[:, 0] = wrap_angle(err[:, 0])
    rms = np.sqrt((err ** 2).mean(0))
    lsb = np.array([np.pi / 128, 8 / 127])   # int8 state LSBs (DESIGN.md §2)
    print(f"{name}: one-step RMS error  d_theta {rms[0]:.5f} rad ({rms[0] / lsb[0]:.3f} LSB), "
          f"d_omega {rms[1]:.5f} rad/s ({rms[1] / lsb[1]:.3f} LSB)")


def main():
    rng = np.random.default_rng(SEED)
    x, y = collect(rng, SEED, N_TRANSITIONS)
    xm, xs = x.mean(0), x.std(0) + 1e-6     # + 1e-6 guards against dividing by 0
    ym, ys = y.mean(0), y.std(0) + 1e-6
    print(f"{len(x)} transitions; input mean {xm}, std {xs}")
    print(f"target mean {ym}, std {ys}")

    layers = train(init_layers(rng), (x - xm) / xs, (y - ym) / ys, rng)

    # the hidden scale s_h is calibrated on the largest pre-activation the
    # network produces on its own training data (DESIGN.md §4.1)
    (w1, b1), (w2, b2) = layers
    a1_max = float(((x - xm) / xs @ w1 + b1).max())
    print(f"largest layer-1 pre-activation on the training data: {a1_max:.4f}")

    # held-out data from a different seed, to check the model generalizes
    x_test, y_test = collect(np.random.default_rng(SEED + 1), SEED + 1, 5_000)
    rmse_report("training data", layers, x, y, xm, xs, ym, ys)
    rmse_report("held-out data", layers, x_test, y_test, xm, xs, ym, ys)

    # store the weights as W[out][in] to match DESIGN.md §4 and the ROM order
    out = ROOT / "model" / "dynamics_fp32.npz"
    np.savez(out, W1=w1.T, b1=b1, W2=w2.T, b2=b2, xm=xm, xs=xs, ym=ym, ys=ys,
             a1_max=np.float32(a1_max))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
