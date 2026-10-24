"""
conic_head.py -- the norm-linear (conic) classification head and its prototype limit.

    g_k(z) = <w_k, z - a_k> + alpha_k * sqrt(||z - a_k||^2 + eps^2)      eq. (1)
    s_k(z) = rho_k - g_k(z)                 logit; the class of smallest g_k - rho_k wins
    alpha_k = tau_k * ||w_k||,   tau_k = tau_floor + softplus(eta_k)

Budget of the head (Definition 1, Theorem 4):
    Lipschitz constant   ||grad_z g_k||  <= ||w_k|| + alpha_k
    curvature            ||Hess_z g_k||   = alpha_k / phi  <= alpha_k / eps
    coercivity           g_k(z)          >= (alpha_k - ||w_k||) ||z - a_k||

Conic type (Theorem 5): the sublevel set is a quadric with spatial matrix
    A_k = alpha_k^2 I - w_k w_k^T,
an ellipsoid for tau_k > 1, a paraboloid for tau_k = 1 and a hyperboloid for tau_k < 1.

    python3 src/conic_head.py          runs the self-tests (seconds, CPU, no data)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConicHead(nn.Module):
    """Norm-linear conic classification head, eq. (1)."""

    def __init__(self, dim: int, num_classes: int, eps: float = 0.1,
                 tau_init: float = 1.5, learn_tau: bool = True,
                 learn_rho: bool = True, tau_floor: float = 0.0):
        super().__init__()
        # tau = tau_floor + softplus(eta). tau_floor = 1 keeps every class region an
        # ellipsoid (Table 5, "tau >= 1").
        assert tau_init > tau_floor, 'tau_init must exceed tau_floor'
        self.dim, self.num_classes, self.eps = dim, num_classes, eps
        self.tau_floor = tau_floor
        self.a = nn.Parameter(torch.randn(num_classes, dim) * 0.1)               # apices
        self.w = nn.Parameter(torch.randn(num_classes, dim) / math.sqrt(dim))    # axes
        eta0 = math.log(math.expm1(tau_init - tau_floor))    # softplus(eta0) = tau_init - floor
        self.eta = nn.Parameter(torch.full((num_classes,), eta0), requires_grad=learn_tau)
        self.rho = nn.Parameter(torch.zeros(num_classes), requires_grad=learn_rho)

    # ---- derived quantities -------------------------------------------------
    def tau(self) -> torch.Tensor:
        return self.tau_floor + F.softplus(self.eta)      # (K,)

    def alpha(self) -> torch.Tensor:
        return self.tau() * self.w.norm(dim=1)            # (K,)

    def budget(self) -> dict:
        """The three quantities of Definition 1, in closed form."""
        with torch.no_grad():
            wn, al = self.w.norm(dim=1).detach(), self.alpha().detach()
        return {
            "L_g_per_class": wn + al,                     # ||grad_z g_k||
            "L_g_bound": math.sqrt(self.num_classes) * float((wn + al).max()),
            "beta_g": float((al / self.eps).max()),       # ||Hess_z g_k||
            "kappa_g_per_class": al - wn,                 # coercivity, > 0 iff tau > 1
            "tau": self.tau().detach(),
        }

    # ---- forward ------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, d) -> logits (N, K).

        Expanded so that no (N, K, d) tensor is built:
            <w_k, z - a_k>   = z.w_k - a_k.w_k
            ||z - a_k||^2    = ||z||^2 - 2 z.a_k + ||a_k||^2
        """
        lin = z @ self.w.t() - (self.a * self.w).sum(1).unsqueeze(0)     # (N,K)
        sq = ((z * z).sum(1, keepdim=True)
              - 2.0 * (z @ self.a.t())
              + (self.a * self.a).sum(1).unsqueeze(0))                   # (N,K)
        phi = torch.sqrt(sq.clamp_min(0.0) + self.eps ** 2)              # clamp: cancellation
        return self.rho.unsqueeze(0) - (lin + self.alpha().unsqueeze(0) * phi)

    # ---- closed-form curvature diagnostic ----------------------------------
    @torch.no_grad()
    def head_curvature(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Spectral norm of  sum_k (dl/ds_k) Hess_z s_k  per sample, in closed form.

        Hess_z s_k = -alpha_k (I/phi_k - u_k u_k^T / phi_k^3), so the sum is a
        rank-(<=K) update of a scaled identity and needs no autograd.
        Returns (N,). Cost O(N d K^2).
        """
        logits = self.forward(z)
        p = torch.softmax(logits, dim=1)
        dl = p.clone()
        dl.scatter_add_(1, target.view(-1, 1), -torch.ones_like(target, dtype=p.dtype).view(-1, 1))
        u = z.unsqueeze(1) - self.a.unsqueeze(0)                     # (N,K,d)
        phi = torch.sqrt((u * u).sum(-1) + self.eps ** 2)            # (N,K)
        al = self.alpha().unsqueeze(0)                               # (1,K)
        coef = -dl * al                                              # (N,K)
        iso = (coef / phi).sum(1)                                    # (N,) scalar part
        S = torch.diag_embed(coef / phi ** 3)                        # (N,K,K)
        # M = U^T S U has rank <= K. With U^T = Q R (thin QR), the nonzero spectrum
        # of M is that of the symmetric K x K matrix R S R^T.
        Q, R = torch.linalg.qr(u.transpose(1, 2))                    # (N,d,K),(N,K,K)
        nu = torch.linalg.eigvalsh(R @ S @ R.transpose(1, 2))        # (N,K)
        cand = (iso.unsqueeze(1) - nu).abs()                         # eigenvalues of H
        if self.dim > self.num_classes:                              # iso survives on U^perp
            cand = torch.cat([cand, iso.abs().unsqueeze(1)], dim=1)
        return cand.max(dim=1).values

    # ---- geometry -----------------------------------------------------------
    @torch.no_grad()
    def signature(self) -> torch.Tensor:
        """Inertia of A_k = alpha_k^2 I - w_k w_k^T as (n_pos, n_zero, n_neg) per class."""
        out = []
        for k in range(self.num_classes):
            A = self.alpha()[k] ** 2 * torch.eye(self.dim, dtype=self.w.dtype) \
                - torch.outer(self.w[k], self.w[k])
            ev = torch.linalg.eigvalsh(A)
            tol = 1e-9 * max(1.0, float(ev.abs().max()))
            out.append([int((ev > tol).sum()), int((ev.abs() <= tol).sum()), int((ev < -tol).sum())])
        return torch.tensor(out)


class ProtoHead(nn.Module):
    """Prototype limit of the norm-linear head: every axis is zero, so

        g_k(z) = alpha_k * sqrt(||z - a_k||^2 + eps^2),   alpha_k = softplus(eta_k).

    The aperture is learned directly, since alpha_k = tau_k ||w_k|| would vanish with
    w_k = 0. By Proposition 7 only the class of strictly smallest aperture can have an
    unbounded cell."""

    def __init__(self, dim: int, num_classes: int, eps: float = 0.1,
                 alpha_init: float = 1.5):
        super().__init__()
        self.dim, self.num_classes, self.eps = dim, num_classes, eps
        self.a = nn.Parameter(torch.randn(num_classes, dim) * 0.1)
        self.eta = nn.Parameter(torch.full((num_classes,), math.log(math.expm1(alpha_init))))
        self.rho = nn.Parameter(torch.zeros(num_classes))

    def alpha(self) -> torch.Tensor:
        return F.softplus(self.eta)

    def budget(self) -> dict:
        with torch.no_grad():
            al = self.alpha().detach()
        return {"L_g_per_class": al, "beta_g": float((al / self.eps).max()),
                "kappa_g_per_class": al,
                "alpha_spread": float((al.max() - al.min()) / al.mean())}

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        sq = ((z * z).sum(1, keepdim=True) - 2.0 * (z @ self.a.t())
              + (self.a * self.a).sum(1).unsqueeze(0))
        phi = torch.sqrt(sq.clamp_min(0.0) + self.eps ** 2)
        return self.rho.unsqueeze(0) - self.alpha().unsqueeze(0) * phi


# ============================== self-tests ==================================

def _analytic_feature_hessian(head: ConicHead, z: torch.Tensor, y: int):
    """G^T H_l G + sum_k (dl/ds_k) Hess_z s_k   for one sample z (d,)."""
    d, K, eps = head.dim, head.num_classes, head.eps
    u = z.unsqueeze(0) - head.a                          # (K,d)
    phi = torch.sqrt((u * u).sum(-1) + eps ** 2)         # (K,)
    al = head.alpha()
    G = -(head.w + al.unsqueeze(1) * u / phi.unsqueeze(1))          # (K,d) = ds_k/dz
    s = head.forward(z.unsqueeze(0)).squeeze(0)
    p = torch.softmax(s, 0)
    Hl = torch.diag(p) - torch.outer(p, p)
    dl = p.clone(); dl[y] -= 1.0
    H = G.T @ Hl @ G
    for k in range(K):
        Hk = -al[k] * (torch.eye(d, dtype=z.dtype) / phi[k]
                       - torch.outer(u[k], u[k]) / phi[k] ** 3)
        H = H + dl[k] * Hk
    return H


def _run_tests():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    d, K, N, eps = 6, 5, 64, 0.12
    head = ConicHead(d, K, eps=eps, tau_init=1.4)
    ok = True
    print("=" * 66)
    print("CONIC HEAD -- SELF-TESTS   (torch %s, float64)" % torch.__version__)
    print("=" * 66)

    # 1. analytic feature Hessian (Lemma 2) against autograd
    z = torch.randn(d, requires_grad=True)
    y = 2
    lossf = lambda zz: F.cross_entropy(head.forward(zz.unsqueeze(0)), torch.tensor([y]))
    Hauto = torch.autograd.functional.hessian(lossf, z)
    Hana = _analytic_feature_hessian(head, z.detach(), y)
    rel = float(((Hauto - Hana).norm() / Hauto.norm()).detach())
    print("1) feature Hessian: ||analytic - autograd|| / ||autograd|| = %.3e" % rel)
    ok &= rel < 1e-10

    # 2. closed-form head-curvature term against autograd
    zb = torch.randn(N, d)
    tb = torch.randint(0, K, (N,))
    hc = head.head_curvature(zb, tb)
    errs = []
    for i in range(3):
        zi = zb[i].clone().requires_grad_(True)
        s = head.forward(zi.unsqueeze(0)).squeeze(0)
        p = torch.softmax(s, 0).detach()
        dl = p.clone(); dl[tb[i]] -= 1.0
        Hsum = torch.zeros(d, d)
        for k in range(K):
            gk = torch.autograd.grad(head.forward(zi.unsqueeze(0))[0, k], zi, create_graph=True)[0]
            Hk = torch.stack([torch.autograd.grad(gk[j], zi, retain_graph=True)[0] for j in range(d)])
            Hsum = Hsum + dl[k] * Hk
        errs.append(abs(float(torch.linalg.matrix_norm(Hsum, ord=2)) - float(hc[i])))
    print("2) head-curvature closed form vs autograd: max abs error = %.3e" % max(errs))
    ok &= max(errs) < 1e-10

    # 3. Lipschitz constant of Theorem 4: observed against predicted
    b = head.budget()
    zs = head.a[0] + (10 ** (torch.rand(4000, 1) * 4 - 3)) * torch.randn(4000, d)
    zs.requires_grad_(True)
    g0 = (zs - head.a[0]) @ head.w[0] + head.alpha()[0] * torch.sqrt(
        ((zs - head.a[0]) ** 2).sum(1) + eps ** 2)
    grads = torch.autograd.grad(g0.sum(), zs)[0]
    obs_L = float(grads.norm(dim=1).max())
    print("3) class 0: sup ||grad g|| observed %.4f <= ||w|| + alpha = %.4f"
          % (obs_L, float(b["L_g_per_class"][0])))
    ok &= obs_L <= float(b["L_g_per_class"][0]) + 1e-12

    # 4. per-sample head curvature stays below 2 beta_g (Lemma 3)
    zb2 = torch.randn(512, d) * 3
    tb2 = torch.randint(0, K, (512,))
    hc2 = head.head_curvature(zb2, tb2)
    print("4) per-sample head curvature: max %.4f <= 2 beta_g = %.4f"
          % (float(hc2.max()), 2 * b["beta_g"]))
    ok &= float(hc2.max()) <= 2 * b["beta_g"]

    # 5. conic type set by tau (Theorem 5)
    print("5) inertia of A_k = alpha^2 I - w w^T against tau:")
    want = {1.8: "ellipsoid", 1.0: "paraboloid", 0.6: "hyperboloid"}
    for tau, name in want.items():
        h = ConicHead(d, 1, eps=eps, tau_init=tau)
        ins = h.signature()[0].tolist()
        kind = "ellipsoid" if ins[1] == 0 and ins[2] == 0 else (
            "paraboloid" if ins[1] > 0 else "hyperboloid")
        print("   tau = %.1f -> inertia (%d,%d,%d)  %s" % (tau, *ins, kind))
        ok &= kind == name

    # 6. the head trains: two Gaussian blobs, 200 steps
    torch.manual_seed(1)
    Xa = torch.randn(300, d) + 2.0
    Xb = torch.randn(300, d) - 2.0
    X = torch.cat([Xa, Xb]); Y = torch.cat([torch.zeros(300), torch.ones(300)]).long()
    h = ConicHead(d, 2, eps=eps, tau_init=1.5)
    opt = torch.optim.SGD(h.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(h(X), Y)
        loss.backward(); opt.step()
    acc = float((h(X).argmax(1) == Y).double().mean())
    print("6) two Gaussian blobs: loss %.4f, accuracy %.3f" % (float(loss.detach()), acc))
    ok &= acc > 0.99
    print("=" * 66)
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _run_tests() else 1)
