# On latent collapse in LeWM

This note explains one failure mode I hit while training LeWM, why it happens, and the fix.
Everything here comes from the code and the runs in this repository.

## Setup

LeWM is a latent world model. A ViT encoder maps a 64x64 top-down view to a latent of
dimension 192. An action-conditioned predictor rolls that latent forward. Planning is done in
the latent space with CEM. To keep the latents well-behaved I use SIGReg from LeJEPA
(Balestriero and LeCun, 2025), which pushes the latent distribution toward an isotropic
Gaussian.

## Why SIGReg alone collapses

SIGReg standardizes the latents before running its Gaussianity test. In `src/sigreg.py` the
latents are normalized per dimension before the test:

    z_std = z.std(dim=0, keepdim=True).clamp(min=1e-6)
    z_norm = (z - z_mean) / z_std

That normalization removes any information about scale. The test is therefore invariant to the
scale of the latents, and a collapsed encoder whose latents have almost no spread passes it
just as well as a healthy one. SIGReg gives no gradient that pushes back on collapse.

The consequence shows up end to end. With nothing else constraining the encoder, it learns to
map every image to almost the same point. The predicted latent and the goal latent then sit on
top of each other for every action sequence, so the planning cost is flat. A flat cost gives
the planner nothing to optimize, and it stops producing useful actions.

## The fix: a variance and covariance term

I add the VICReg variance and covariance terms on top of SIGReg (`src/lewm.py`). The variance
term is a hinge that keeps the per-dimension standard deviation above a floor:

    std = torch.sqrt(z_for_sig.var(dim=0) + 1e-4)
    loss_var = torch.mean(F.relu(self.cfg.var_gamma - std))

with `var_gamma = 1.0`. The covariance term decorrelates the latent dimensions by pushing the
off-diagonal entries of the latent covariance matrix to zero. In the config these are weighted
`var_weight = 1.0` and `cov_weight = 0.04`.

The variance hinge is what directly blocks collapse. It makes a low-variance latent expensive,
so the encoder can no longer shrink everything to a point.

## What changes

Before the fix, the per-dimension standard deviation of the latents sits around 0.05, the
collapsed regime. After adding the variance and covariance term it moves to about 1.0 and
stays there through training, and the planner produces useful actions again. This is an
observation read off the latents during training. It is not logged as a per-epoch curve in the
current training log, so the two numbers should be read as the collapsed and recovered regimes,
not as a measured trajectory.

## The remaining limit

Fixing collapse makes the model usable but does not close the gap with the analytic baseline.

The analytic controller (bounded-curvature Dubins path, pure pursuit, final parking step) is a
privileged reference: it reads the true goal pose from the environment. It parks with 100%
strict success over 10 seeds, about 1.7 cm from the center of the spot and about 0.9 degrees
off axis, with no collision.

The learned latent planner drives the car toward the spot but does not reliably reach that
precise alignment. The most likely cause is the size of the training set, about 2250 episodes,
which is modest for a model that has to predict fine-grained end-of-maneuver states. Closing
this gap is the next step: more data, and a predictor that is more accurate near the goal.
