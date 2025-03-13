
import numbers
import warnings

from collections.abc import Sequence
from types import ModuleType

import numpy as np
import pytensor.tensor as at

import pymc as pm

from pymc.gp.cov import Covariance
from pymc.gp.gp import Base
from pymc.gp.mean import Mean, Zero

TensorLike = np.ndarray | at.TensorVariable
from pymc.gp.hsgp_approx import approx_hsgp_hyperparams
from pymc.gp.cov import Matern52, Matern32, Exponential
from CovarianceSphericalMetric import Matern32Chordal, Matern52Chordal, ExponentialChordal
import scipy.sparse as sp
from pytensor import sparse
#This is a subclass of the Gaussian Process (GP) Base class in PYMC.
#The HSGP_latlon class is a modification of the Hilbert Space approximation GP in PYMC
#The HSGP_laton class is intended to be used as a latent process where the inputs are lat lon coordinates



def lonlat2xyz(lonlat):
    '''Function that converts lat lon coordinates to Cartesian coordinates, 
    where r is the radius of the earth '''
    lonlat = np.deg2rad(lonlat)
    r=6378.137
    return r * at.stack(
        [
            at.cos(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
            at.sin(lonlat[..., 0]) * at.cos(lonlat[..., 1]),
            at.sin(lonlat[..., 1]),
        ],
        axis=-1,
    )
def set_boundary(X: TensorLike, c: numbers.Real | TensorLike) -> np.ndarray:
    """Set the boundary using `X` and `c`.  `X` can be centered around zero but doesn't have to be,
    and `c` is usually a scalar multiplier greater than 1.0, but it may also be one value per
    dimension or column of `X`.
    """
    # compute radius. Works whether X is 0-centered or not
    X = lonlat2xyz(X)
    S = (at.max(X, axis=0) - at.min(X, axis=0)) / 2.0

    L = (c * S).eval()  # eval() makes sure L is not changed with out-of-sample preds
    return L


def calc_eigenvalues(L: TensorLike, m: Sequence[int]):
    """Calculate eigenvalues of the Laplacian."""

    S = np.meshgrid(*[np.arange(1, 1 + m[d]) for d in range(len(m))])
    S_arr = np.vstack([s.flatten() for s in S]).T

    return np.square((np.pi * S_arr) / (2 * L))


def calc_eigenvectors(
    Xs: TensorLike,
    L: TensorLike,
    eigvals: TensorLike,
    m: Sequence[int],
):
    """Calculate eigenvectors of the Laplacian. These are used as basis vectors in the HSGP
    approximation.
    """
    Xs = lonlat2xyz(Xs)
    m_star = int(np.prod(m))

    phi = at.ones((Xs.shape[0], m_star))
    for d in range(len(m)):
        c = 1.0 / at.sqrt(L[d])
        term1 = at.sqrt(eigvals[:, d])
        term2 = at.tile(Xs[:, d][:, None], m_star) + L[d]
        phi *= c * at.sin(term1 * term2)

    return phi

def calculate_Kapprox(Xs, c, m, chosen_ell, cov_func_name="matern52"):
    '''Function that computes an approximation of a given covariance function,
     by taking in a (lat,lon) coordinate array, a float c, a list
    of integers m with length 3, the length scale used for the actual covariance function
    and the covariance function name. The function returns
     the Hilbert Space approximated covariance function (Operator). '''
    # Calculate Phi and the diagonal matrix of power spectral densities
    cov_func_name = cov_func_name.lower()
    cov_func_xyz_dict = {"matern32":Matern32, "matern52":Matern52, "exponential":Exponential}
    cov_func_xyz = cov_func_xyz_dict[cov_func_name](3, chosen_ell)
    XYZ_region = lonlat2xyz(Xs)
    XYZ_region_centered =(at.max(XYZ_region, axis=0)+ at.min(XYZ_region, axis=0))/2
    XYZ =XYZ_region-XYZ_region_centered
    L = pm.gp.hsgp_approx.set_boundary(XYZ, c)
    eigvals = pm.gp.hsgp_approx.calc_eigenvalues(L, m)
    phi = pm.gp.hsgp_approx.calc_eigenvectors(XYZ, L, eigvals, m)
    omega = at.sqrt(eigvals)
    psd =  cov_func_xyz.power_spectral_density(omega)
    return (phi @ at.diag(psd) @ phi.T).eval()

def calculate_K(Xs,chosen_ell, cov_func_name="matern52"):
    '''Function that computes the covariance matrix using Chordal distance, given a 
    (lat,lon) coordinate array, a length scale and the covariance function name'''
    cov_func_name = cov_func_name.lower()
    cov_func_chordal_dict = {"matern32":Matern32Chordal, "matern52":Matern52Chordal, "exponential":ExponentialChordal}
    cov_func = cov_func_chordal_dict[cov_func_name](input_dims=2, ls=chosen_ell )
    return cov_func(Xs).eval()

def approx_hsgp_hyperparams_xyz(XY_region,  lengthscale, variable_name, cov_func_name="matern52"):
    '''Function that return the smallest recommended number of basis vectors, respectivley, scaling factor.
    Taken over the region contaning the coordinates for in and out of sample predictions, the lengthscale used 
    the variable name for w.r.t. to the length scale and name of covariance function.'''
    cov_func_name = cov_func_name.lower()
    lengthscale_range=[lengthscale - 10, lengthscale + 10]
    XYZ_region = lonlat2xyz(XY_region)
    XYZ_region_centered =(at.max(XYZ_region, axis=0)+ at.min(XYZ_region, axis=0))/2
    XYZs_region =XYZ_region-XYZ_region_centered
    XYZs_max = [at.max(XYZs_region[:,i], axis=0).eval() for i in range(3)]
    m_and_c_for_xyz = [pm.gp.hsgp_approx.approx_hsgp_hyperparams(x_range=[-XYZs_max[i], XYZs_max[i]], 
                                                                 lengthscale_range=lengthscale_range, 
                                                                 cov_func=cov_func_name) for i in range(3)]
    m = [ m_and_c_for_xyz[i][0] for i in range(3)]
    c = [ np.round(m_and_c_for_xyz[i][1],1) for i in range(3)]
    print(f"For the variable {variable_name}")
    print("Recommended smallest number of basis vectors [Xs,Ys,Zs] (m):", m)
    print("Recommended smallest scaling factor [Xs,Ys,Zs] (c):", c)


class HSGP_LatLon(Base):
    """
    Hilbert Space Gaussian process approximation.

    The `gp.HSGP` class is an implementation of the Hilbert Space Gaussian process.  It is a
    reduced rank GP approximation that uses a fixed set of basis vectors whose coefficients are
    random functions of a stationary covariance function's power spectral density.  Its usage
    is largely similar to `gp.Latent`.  Like `gp.Latent`, it does not assume a Gaussian noise model
    and can be used with any likelihood, or as a component anywhere within a model.  Also like
    `gp.Latent`, it has `prior` and `conditional` methods.  It supports any sum of covariance
    functions that implement a `power_spectral_density` method. (Note, this excludes the
    `Periodic` covariance function, which uses a different set of basis functions for a
    low rank approximation, as described in `HSGPPeriodic`.).

    For information on choosing appropriate `m`, `L`, and `c`, refer to Ruitort-Mayol et al. or to
    the PyMC examples that use HSGP.

    To work with the HSGP in its "linearized" form, as a matrix of basis vectors and a vector of
    coefficients, see the method `prior_linearized`.

    The HSGP_LatLon converts the LatLon coordinates to cartesian coordinates and computes the 
    eigenvalues, resp, -vectors in the new coordinate system, which is used to approximate the 
    covariance function. Recall that the covariance function uses the chordal distance function 
    as a metric. 

    Parameters
    ----------
    m: list
        The number of basis vectors to use for each active dimension (covariance parameter
        `active_dim` +1 ).
    L: list
        The boundary of the space for each `active_dim`.  It is called the boundary condition.
        Choose L such that the domain `[-L, L]` contains all points in the column of X given by the
        `active_dim` + 1.
    c: float
        The proportion extension factor.  Used to construct L from X.  Defined as `S = max|X|` such
        that `X` is in `[-S, S]`.  `L` is calculated as `c * S`.  One of `c` or `L` must be
        provided.  Further information can be found in Ruitort-Mayol et al.
    drop_first: bool
        Default `False`. Sometimes the first basis vector is quite "flat" and very similar to
        the interceat term.  When there is an interceat in the model, ignoring the first basis
        vector may improve sampling. This argument will be deprecated in future versions.
    parametrization: str
        Whether to use the `centered` or `noncentered` parametrization when multiplying the
        basis by the coefficients.
    ls: Tensorlike
        The ls is the length scale used for the cooresponding cov functions, this can either by
        a float or some pymc distribution. Should be the same as for the covariance function. 
    eps: Tensorlike
        The eps is the error term used for the cooresponding cov functions, this can either by
        a float or some pymc distribution. Should be the same as for the covariance function.
    cov_func_name: str
        The name of the covariance function being used. 
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.
    cov_func: Covariance function, must be an instance of `Stationary` and implement a
        `power_spectral_density` method.    


    References
    ----------
    -   Ruitort-Mayol, G., and Anderson, M., and Solin, A., and Vehtari, A. (2022). Practical
        Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming

    -   Solin, A., Sarkka, S. (2019) Hilbert Space Methods for Reduced-Rank Gaussian Process
        Regression.
    """

    def __init__(
        self,
        m: Sequence[int],
        L: Sequence[float] | None = None,
        c: numbers.Real | None = None,
        drop_first: bool = False,
        parametrization: str | None = "noncentered",
        *,
        ls: float,
        eps: TensorLike,
        cov_func_name: str,   
        mean_func: Mean = Zero(),
        cov_func: Covariance,
    ):
        arg_err_msg = (
            "`m` and `L`, if provided, must be sequences with one element per active "
            "dimension of the kernel or covariance function."
        )

        if not isinstance(m, Sequence):
            raise ValueError(arg_err_msg)

        if len(m) != cov_func.n_dims + 1:
            raise ValueError(arg_err_msg)
        m = tuple(m)

        if (L is None and c is None) or (L is not None and c is not None):
            raise ValueError("Provide one of `c` or `L`")

        if L is not None and (not isinstance(L, Sequence) or len(L) != cov_func.n_dims +1):
            raise ValueError(arg_err_msg)

        if L is None and c is not None and c < 1.2:
            warnings.warn("For an adequate approximation `c >= 1.2` is recommended.")

        if parametrization is not None:
            parametrization = parametrization.lower().replace("-", "").replace("_", "")

        if parametrization not in ["centered", "noncentered"]:
            raise ValueError("`parametrization` must be either 'centered' or 'noncentered'.")

        if drop_first:
            warnings.warn(
                "The drop_first argument will be deprecated in future versions."
                " See https://github.com/pymc-devs/pymc/pull/6877",
                DeprecationWarning,
            )

        self._drop_first = drop_first
        self._m = m
        self._m_star = self.n_basis_vectors = int(np.prod(self._m))
        self._L: at.TensorVariable | None = None
        if L is not None:
            self._L = at.as_tensor(L).eval()  # make sure L cannot be changed
        self._c = c
        self._ls = ls
        self._eps = eps
        self._cov_func_name = cov_func_name.lower()
        self._parametrization = parametrization
        self._X_center = None

        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        raise NotImplementedError("Additive HSGPs aren't supported.")

    @property
    def L(self) -> at.TensorVariable:
        if self._L is None:
            raise RuntimeError("Boundaries `L` required but still unset.")
        return self._L

    @L.setter
    def L(self, value: TensorLike):
        self._L = at.as_tensor_variable(value)

    def prior_linearized(self, X: TensorLike):
        """Linearized version of the HSGP.  Returns the Laplace eigenfunctions and the square root
        of the power spectral density needed to create the GP.

        This function allows the user to bypass the GP interface and work with the basis
        and coefficients directly.  This format allows the user to create predictions using
        `pm.set_data` similarly to a linear model.  It also enables computational speed ups in
        multi-GP models, since they may share the same basis.  The return values are the Laplace
        eigenfunctions `phi`, and the square root of the power spectral density.
        An example is given below.

        Parameters
        ----------
        X: array-like
            Function input values.

        Returns
        -------
        phi: array-like
            Either Numpy or PyTensor 2D array of the fixed basis vectors.  There are n rows, one
            per row of `Xs` and `prod(m)` columns, one for each basis vector.
        sqrt_psd: array-like
            Either a Numpy or PyTensor 1D array of the square roots of the power spectral
            densities.

        Examples
        --------
        .. code:: python

            # A one dimensional column vector of inputs.
            X = np.linspace(0, 10, 100)[:, None]

            with pm.Model() as model:
                eta = pm.Exponential("eta", lam=1.0)
                ell = pm.InverseGamma("ell", mu=5.0, sigma=5.0)
                cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)

                # m = [200] means 200 basis vectors for the first dimension
                # L = [10] means the approximation is valid from Xs = [-10, 10]
                gp = pm.gp.HSGP(m=[200], L=[10], cov_func=cov_func)

                # Set X as Data so it can be mutated later, and then pass it to the GP
                X = pm.Data("X", X)
                phi, sqrt_psd = gp.prior_linearized(X=X)

                # Specify standard normal prior in the coefficients, the number of which
                # is given by the number of basis vectors, saved in `n_basis_vectors`.
                beta = pm.Normal("beta", size=gp.n_basis_vectors)

                # The (non-centered) GP approximation is given by:
                f = pm.Deterministic("f", phi @ (beta * sqrt_psd))

                # The centered approximation can be more efficient when
                # the GP is stronger than the noise
                # beta = pm.Normal("beta", sigma=sqrt_psd, size=gp.n_basis_vectors)
                # f = pm.Deterministic("f", phi @ beta)

                ...


            # Then it works just like a linear regression to predict on new data.
            # First mutate the data X,
            x_new = np.linspace(-10, 10, 100)
            with model:
                pm.set_data({"X": x_new[:, None]})

            # and then make predictions for the GP using posterior predictive sampling.
            with model:
                ppc = pm.sample_posterior_predictive(idata, var_names=["f"])
        """
        # Important: fix the computation of the midpoint of X.
        # If X is mutated later, the training midpoint will be subtracted, not the testing one.
        if self._X_center is None:
            self._X_center = (at.max(X, axis=0) + at.min(X, axis=0)).eval() / 2
        Xs = X - self._X_center  # center for accurate computation

        # Index Xs using input_dim and active_dims of covariance function
        Xs, _ = self.cov_func._slice(Xs)

        # If not provided, use Xs and c to set L
        if self._L is None:
            assert isinstance(self._c, numbers.Real | np.ndarray | at.TensorVariable)
            self.L = at.as_tensor(set_boundary(Xs, self._c))  # Xs should be 0-centered
        else:
            self.L = self._L
        
        cov_func_xyz_dict = {"matern32":Matern32, "matern52":Matern52, "exponential":Exponential}

        eigvals = calc_eigenvalues(self.L, self._m)
        phi = calc_eigenvectors(Xs, self.L, eigvals, self._m)
        omega = at.sqrt(eigvals)
        cov_func_xyz = self._eps**2 * cov_func_xyz_dict[self._cov_func_name](3, ls=self._ls) 
        psd = cov_func_xyz.power_spectral_density(omega)

        i = int(self._drop_first is True)
        return phi[:, i:], at.sqrt(psd[i:])

    def prior(
        self,
        name: str,
        X: TensorLike,
        hsgp_coeffs_dims: str | None = None,
        gp_dims: str | None = None,
        *args,
        **kwargs,
    ):  # type: ignore
        """
        Returns the (approximate) GP prior distribution evaluated over the input locations `X`.
        For usage examples, refer to `pm.gp.Latent`.

        Parameters
        ----------
        name: str
            Name of the random variable
        X: array-like
            Function input values.
        hsgp_coeffs_dims: str, default None
            Dimension name for the HSGP basis vectors.  
        gp_dims: str, default None
            Dimension name for the GP random variable.
        """
        phi, sqrt_psd = self.prior_linearized(X)

        if self._parametrization == "noncentered":
            self._beta = pm.Normal(
                f"{name}_hsgp_coeffs_",
                size=self._m_star - int(self._drop_first),
                dims=hsgp_coeffs_dims,
            )
            self._sqrt_psd = sqrt_psd
            f = self.mean_func(X) +  phi @ (self._beta * self._sqrt_psd)

        elif self._parametrization == "centered":
            self._beta = pm.Normal(f"{name}_hsgp_coeffs_", sigma=sqrt_psd, dims=hsgp_coeffs_dims)
            f = self.mean_func(X) + phi @ self._beta

        self.f = pm.Deterministic(name, f, dims=gp_dims)
        return self.f

    def _build_conditional(self, Xnew):
        try:
            beta, X_center = self._beta, self._X_center

            if self._parametrization == "noncentered":
                sqrt_psd = self._sqrt_psd

        except AttributeError:
            raise ValueError(
                "Prior is not set, can't create a conditional.  Call `.prior(name, X)` first."
            )

        Xnew, _ = self.cov_func._slice(Xnew)
        eigvals = calc_eigenvalues(self.L, self._m)
        phi = calc_eigenvectors(Xnew - X_center, self.L, eigvals, self._m)
        i = int(self._drop_first is True)

        if self._parametrization == "noncentered":
            return self.mean_func(Xnew) + phi[:, i:] @ (beta * sqrt_psd)

        elif self._parametrization == "centered":
            return self.mean_func(Xnew) + phi[:, i:] @ beta

    def conditional(self, name: str, Xnew: TensorLike, dims: str | None = None):  # type: ignore
        """
        Returns the (approximate) conditional distribution evaluated over new input locations
        `Xnew`.

        Parameters
        ----------
        name
            Name of the random variable
        Xnew : array-like
            Function input values.
        dims: None
            Dimension name for the GP random variable.
        """
        fnew = self._build_conditional(Xnew)
        return pm.Deterministic(name, fnew, dims=dims)
