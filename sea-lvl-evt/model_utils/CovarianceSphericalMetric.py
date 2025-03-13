import numpy as np
import pymc as pm
import pytensor.tensor as at
from typing import Union
from pytensor.tensor.variable import TensorVariable
TensorLike = Union[np.ndarray, TensorVariable]


class Matern32Chordal(pm.gp.cov.Stationary):
    '''Class the computes the Matern32 covariance under Chordal coordinates  '''
    def __init__(self, input_dims, ls, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Chordal distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r
    
    def _ls(self):
        return self.ls

    def lonlat2xyz(self, lonlat):
        lonlat = np.deg2rad(lonlat)
        return self.r * at.stack(
            [
                at.cos(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 1]),
            ],
            axis=-1,
        )

    def chordal_dist(self, X, Xs=None):
        if Xs is None:
            Xs = X
        X, Xs = at.broadcast_arrays(
            self.lonlat2xyz(X[..., :, None, :]), self.lonlat2xyz(Xs[..., None, :, :])
        )
        return at.sqrt(at.sum(((X - Xs) / self.ls) ** 2, axis=-1) + 1e-12)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.chordal_dist(X, Xs)
        return (1.0 + np.sqrt(3.0) * r) * at.exp(-np.sqrt(3.0) * r)
    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        r"""
        The power spectral density for the Matern32 kernel is:

        .. math::

            S(\boldsymbol\omega) =
                \frac{2^D \pi^{D/2} \Gamma\left(\frac{D+3}{2}\right) 3^{3/2}}
                     {\frac{1}{2}\sqrt{\pi}}
               \prod_{i=1}^{D}\ell_{i}
               \left(3 + \sum_{i=1}^{D}\ell_{i}^2 \boldsymbol\omega_{i}^{2}\right)^{-\frac{D+3}{2}}
        """
        ls = at.ones(self.n_dims) * self.ls
        D32 = (self.n_dims + 3) / 2
        num = (
            at.power(2, self.n_dims)
            * at.power(np.pi, self.n_dims / 2)
            * at.gamma(D32)
            * at.power(3, 3 / 2)
        )
        den = 0.5 * at.sqrt(np.pi)
        pow = at.power(3.0 + at.dot(at.square(omega), at.square(ls)), -1 * D32)
        return (num / den) * at.prod(ls) * pow

class Matern52Chordal(pm.gp.cov.Stationary):
    '''Class the computes the Matern52 covariance under Chordal coordinates  '''
    def __init__(self, input_dims, ls, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Chordal distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r

    def lonlat2xyz(self, lonlat):
        lonlat = np.deg2rad(lonlat)
        return self.r * at.stack(
            [
                at.cos(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 1]),
            ],
            axis=-1,
        )

    def chordal_dist(self, X, Xs=None):
        if Xs is None:
            Xs = X
        X, Xs = at.broadcast_arrays(
            self.lonlat2xyz(X[..., :, None, :]), self.lonlat2xyz(Xs[..., None, :, :])
        )
        return at.sqrt(at.sum(((X - Xs) / self.ls) ** 2, axis=-1) + 1e-12)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.chordal_dist(X, Xs)
        return (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * at.pow(r, 2)) * at.exp(-1.0 * np.sqrt(5.0) * r)
    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        r"""
        The power spectral density for the Matern52 kernel is:

        .. math::

           S(\boldsymbol\omega) =
               \frac{2^D \pi^{\frac{D}{2}} \Gamma(\frac{D+5}{2}) 5^{5/2}}
                    {\frac{3}{4}\sqrt{\pi}}
               \prod_{i=1}^{D}\ell_{i}
               \left(5 + \sum_{i=1}^{D}\ell_{i}^2 \boldsymbol\omega_{i}^{2}\right)^{-\frac{D+5}{2}}
        """
        ls = at.ones(self.n_dims) * self.ls
        D52 = (self.n_dims + 5) / 2
        num = (
            at.power(2, self.n_dims)
            * at.power(np.pi, self.n_dims / 2)
            * at.gamma(D52)
            * at.power(5, 5 / 2)
        )
        den = 0.75 * at.sqrt(np.pi)
        pow = at.power(5.0 + at.dot(at.square(omega), at.square(ls)), -1 * D52)
        return (num / den) * at.prod(ls) * pow


class ExponentialChordal(pm.gp.cov.Stationary):
    """
    The Exponential kernel with Chordal distance

    .. math::

       k(x, x') = \mathrm{exp}\left[ -\frac{||x - x'||}{2\ell} \right]
    """
    def __init__(self, input_dims, ls, r=6378.137, active_dims=None):
        if input_dims != 2:
            raise ValueError("Chordal distance is only defined on 2 dimensions")
        super().__init__(input_dims, ls=ls, active_dims=active_dims)
        self.r = r

    def lonlat2xyz(self, lonlat):
        lonlat = np.deg2rad(lonlat)
        return self.r * at.stack(
            [
                at.cos(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
                at.sin(lonlat[..., 1]),
            ],
            axis=-1,
        )

    def chordal_dist(self, X, Xs=None):
        if Xs is None:
            Xs = X
        X, Xs = at.broadcast_arrays(
            self.lonlat2xyz(X[..., :, None, :]), self.lonlat2xyz(Xs[..., None, :, :])
        )
        return at.sqrt(at.sum(((X - Xs) / self.ls) ** 2, axis=-1) + 1e-12)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.chordal_dist(X, Xs)
        return at.exp(-0.5 * r)
    
    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        r"""
        The power spectral density for the ExpQuad kernel is:

        .. math::

           S(\boldsymbol\omega) =
               (\sqrt(2 \pi)^D \prod_{i}^{D}\ell_i
                \exp\left( -\frac{1}{2} \sum_{i}^{D}\ell_i^2 \omega_i^{2} \right)
        """
        ls = at.ones(self.n_dims) * self.ls
        c = at.power(at.sqrt(2.0 * np.pi), self.n_dims)
        exp = at.exp(-0.5 * at.dot(at.square(omega), at.square(ls)))
        return c * at.prod(ls) * exp
