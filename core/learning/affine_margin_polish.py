"""Refine an affine repair in feature coordinates, not squared Gram geometry."""

import numpy as np


def polish_feature_dual(normals, target, initial, *, max_iterations):
    """Return primal and dual proposals; the caller must independently verify.

    A thin QR retains the constraint row space. Active equality systems use
    its singular values directly, avoiding the condition-number squaring of
    a Gram-system solve. Negative multipliers leave the active set; violated
    inactive constraints enter. This search does not prove infeasibility.
    """
    a, b, lam = (np.asarray(value, dtype=np.float64) for value in (normals, target, initial))
    if (a.ndim != 2 or not all(a.shape) or b.shape != (len(a),) or lam.shape != b.shape
            or not all(np.all(np.isfinite(value)) for value in (a, b, lam))
            or type(max_iterations) is not int or max_iterations < 1):
        raise ValueError("invalid feature-space margin polish")
    basis, triangular = np.linalg.qr(a.T, mode="reduced")
    compressed = triangular.T
    lam = np.maximum(lam, 0.).copy()
    active = set(np.flatnonzero(lam > 0.))
    point = np.zeros(compressed.shape[1])
    for _iteration in range(max_iterations):
        if active:
            indices = sorted(active)
            u, singular, vh = np.linalg.svd(compressed[indices], full_matrices=False)
            threshold = np.finfo(float).eps * max(compressed[indices].shape) * singular[0]
            retained = singular > threshold
            projected = u[:, retained].T @ b[indices]
            point = vh[retained].T @ (projected / singular[retained])
            residual = b[indices] - u[:, retained] @ projected
            roundoff = (64 * np.finfo(float).eps * max(compressed[indices].shape)
                        * max(1., np.linalg.norm(b[indices])))
            if np.linalg.norm(residual) > roundoff:
                # Inconsistent active equalities have a left-null direction r.
                # A.T*r = 0, while b.T*r = ||r||^2 > 0: move uphill in the
                # dual until a multiplier reaches zero, then remove that face.
                negative = np.flatnonzero(residual < -roundoff)
                if not len(negative):
                    break
                leaving = min(negative, key=lambda i: lam[indices[i]] / -residual[i])
                fraction = lam[indices[leaving]] / -residual[leaving]
                lam[indices] = np.maximum(lam[indices] + fraction * residual, 0.)
                lam[indices[leaving]] = 0.
                active.remove(indices[leaving])
                continue
            proposal = np.zeros_like(lam)
            proposal[indices] = u[:, retained] @ (projected / singular[retained] ** 2)
            negative = [index for index in indices if proposal[index] < 0.]
            if negative:
                leaving = min(negative, key=lambda i: lam[i] / (lam[i] - proposal[i]))
                fraction = lam[leaving] / (lam[leaving] - proposal[leaving])
                lam = np.maximum(lam + fraction * (proposal - lam), 0.)
                lam[leaving] = 0.
                active.remove(leaving)
                continue
            lam = proposal
        else:
            point = np.zeros_like(point)
            lam = np.zeros_like(lam)
        residual = b - compressed @ point
        missing = [index for index in range(len(lam)) if index not in active and residual[index] > 0.]
        if not missing:
            break
        active.add(max(missing, key=lambda i: residual[i]))
    return basis @ point, lam, _iteration + 1
