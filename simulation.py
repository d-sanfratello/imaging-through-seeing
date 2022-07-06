import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from figaro.mixture import DPGMM, HDPGMM
from figaro.utils import get_priors, plot_multidim
from scipy.stats import multivariate_normal as mn
from tqdm import tqdm


class Planets:
    def __init__(self, mu, cov, weight, rng=None,
                 mu_psf=np.zeros(2), cov_psf=np.zeros((2, 2))):
        self.mu = [m+mu_psf for m in mu]
        self.cov = [c+cov_psf for c in cov]
        self.weight = weight
        self.n_stars = len(mu)

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def rvs(self, size=1):
        """
        from figaro.mixture.mixture.rvs
        """

        idx = self.rng.choice(np.arange(self.n_stars),
                              p=self.weight,
                              size=size)
        ctr = Counter(idx)
        samples = np.empty(shape=(1, 2))
        for i, n in zip(ctr.keys(), ctr.values()):
            samples = np.concatenate((samples,
                                      np.atleast_2d(mn(
                                          self.mu[i],
                                          self.cov[i]).rvs(size))))

        return np.array(samples[1:])


class Simulation:
    def __init__(self, seed, size=100):
        self.size = size
        self.rng = np.random.default_rng(seed=seed)

        self.n_planets = None
        self.planets = None
        self.means = None
        self.sigmas = None
        self.weights = None
        self.convolved_planets = None
        self.n_frames = None

    def initialize_planets(self, means, sigmas, w=[1]):
        self.means = np.array(means)
        self.sigmas = np.array(sigmas)
        self.weights = np.array(w)

        self.n_planets = len(self.means)

    def generate_psfs(self, mu_mean, mu_sigma, s_mean, s_sigma, n_frames):
        self.n_frames = n_frames

        psf_m = self.rng.normal(mu_mean, mu_sigma, size=(n_frames, 2))
        psf_s = abs(self.rng.normal(s_mean, s_sigma, size=n_frames))

        m = self.means
        cov = [s**2 * np.identity(2) for s in self.sigmas]

        self.convolved_planets = np.array(
            [Planets(mu=m,
                     cov=cov,
                     weight=self.weights,
                     rng=self.rng,
                     mu_psf=m_,
                     cov_psf=(s_**2) * np.identity(2))
             for m_, s_ in zip(psf_m, psf_s)])

        return psf_m, psf_s

    def generate_photons(self, n_photons):
        n_photons_planet = n_photons // len(self.means)

        c_planets = self.convolved_planets

        random_samples = np.array([
            np.floor(p.rvs(size=n_photons_planet)) for p in c_planets
        ])

        photons = []

        for img in random_samples:
            image = []
            for px in img:
                if (px[0] < 0 or px[0] >= self.size
                        or px[1] < 0 or px[1] >= self.size):
                    pass
                else:
                    image.append(px)
            photons.append(np.array(image,
                                    dtype=int).reshape((len(image), 2)))

        return photons

    def generate_images(self, photons):
        images = []

        for ph_list in photons:
            img = np.zeros(shape=(self.size, self.size))

            for x, y in ph_list:
                img[x, y] = 1

            images.append(img)

        return images


if __name__ == "__main__":
    sim = Simulation(0000, size=200)

    sim.initialize_planets(means=[[100, 90],
                                  [110, 90]],
                           sigmas=[5, 2],
                           w=[0.5, 0.5])

    m, s = sim.generate_psfs(mu_mean=0,
                             mu_sigma=5,
                             s_mean=5,
                             s_sigma=2,
                             n_frames=50)

    photons = sim.generate_photons(100)
    images = sim.generate_images(photons)

    frames = []
    plt.close(1000)
    fig = plt.figure(1000)
    for n, img in enumerate(images):
        frames.append([plt.imshow(images[n].T,
                                  origin='lower',
                                  cmap=cm.Greys_r,
                                  animated=True)])

    anim = animation.ArtistAnimation(fig,
                                     frames,
                                     interval=40,
                                     blit=True,
                                     repeat=False)
    anim.save('./images/sequence.gif')
    plt.show()

    eps = 1e-3
    bounds = [[0-eps, sim.size-1+eps] for _ in range(2)]
    priors = get_priors(bounds)

    n_draws = 100
    mix = DPGMM(bounds=bounds, prior_pars=priors)
    posteriors = []
    for frame in tqdm(photons, desc='Frames'):
        draws = []
        for _ in range(n_draws):
            draws.append(mix.density_from_samples(frame))
        posteriors.append(draws)

    n_draws_hier = 100
    hier_mix = HDPGMM(bounds)
    hier_draws = []
    for _ in tqdm(range(n_draws_hier)):
        hier_draws.append(hier_mix.density_from_samples(posteriors))

    plot_multidim(hier_draws,
                  show=True,
                  hierarchical=True,
                  true_value=None,
                  save=True,
                  subfolder='images',
                  name='HDPGMM-reconstructed'
                  )
