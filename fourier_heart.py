from functools import partial
import struct
import sys
import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
plt.Circle = partial(plt.Circle, fill=False)

# parametric equation of heart shape
n_points = 1000
t = np.linspace(0.5*np.pi, 2.5*np.pi, n_points)
goal_px = 16 * (np.sin(t))**3 / 16
goal_py = (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) / 16
goal_p = np.array([x + 1j*y for x, y in zip(goal_px, goal_py)])
n_points = len(goal_p)

# fast fourier transform
recipe = np.fft.fft(goal_p)
coeff = recipe / n_points
omegas = n_points * np.fft.fftfreq(n_points)
mask = (np.abs(coeff) > np.quantile(np.abs(coeff), 0.992)) # filter by radii
coeff, omegas = coeff[mask], omegas[mask]
omegas = [x for _, x in sorted(zip(np.abs(coeff), omegas), reverse=True)]
coeff = [x for _, x in sorted(zip(np.abs(coeff), coeff), reverse=True)]

# trace out the final curve first
n_terms = len(coeff)
colors = plt.cm.Reds(np.linspace(0.2, 0.8, n_terms))
t_range = [0.5*np.pi, 2.5*np.pi]
t_grids_fine = np.linspace(*t_range, 100*n_points)
final_p = []
for t in t_grids_fine:
    final_p.append(sum([radius*np.exp(1j*omega*t) for radius, omega in zip(coeff, omegas)]))
final_p = np.array(final_p)

# function for producing gif file
scalar = None
def freeze_shot(t):
    global scalar
    
    fig, ax = plt.subplots(dpi=120, figsize=(3, 3), facecolor='pink')

    # set attributes
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.patch.set_alpha(0.0)
    fig.tight_layout(pad=0.0)

    # compute points for each accumulative fourier sum
    p, accum_p = [], []
    for k in range(n_terms):
        c, omega = coeff[k], omegas[k]
        p.append(c*np.exp(1j * omega * t))
        accum_p.append(sum(p))

    # draw circles
    for k in range(n_terms):
        if k == 0: center = (0, 0)
        else: center = (np.real(accum_p[k-1]), np.imag(accum_p[k-1]))
        ax.add_artist(plt.Circle(center, np.abs(coeff[k]), color=colors[k], linestyle='dotted'))

    # draw points and lines
    ax.plot(np.real([0]+accum_p), np.imag([0]+accum_p), color='gray', alpha=0.5)
    ax.scatter(np.real(accum_p), np.imag(accum_p), color=colors, s=10, zorder=4)
    ax.scatter(0, 0, color=plt.cm.Reds(0.95), s=10, zorder=4)
    smooth_p = final_p[t_grids_fine <= t]
    ax.plot(np.real(smooth_p), np.imag(smooth_p), color=colors[-1], zorder=4)

    # turn matplotlib canvas into numpy array that stores rgb
    fig.canvas.draw() # draw the canvas, cache the renderer
    canvas_shape = fig.canvas.get_width_height()[::-1]
    canvas_bytes = fig.canvas.tostring_rgb()
    image_rgb_1darray = np.frombuffer(canvas_bytes, dtype='uint8')

    # check scaling for the first time
    # somehow different platforms give different canvas dimentions
    # reason unknown(?)
    if scalar is None:
        n_pixels = np.prod(canvas_shape)
        if 3*n_pixels <= len(image_rgb_1darray):
            scalar = np.sqrt((len(image_rgb_1darray) / (3*n_pixels)))
            scalar = int(np.round(scalar))
        else:
            raise Exception('Compression algorithm needed but not implemented yet.')

    # apply scaling and resizing, then return
    canvas_shape = tuple(scalar*dim for dim in canvas_shape)
    image_rgb_2darray = image_rgb_1darray.reshape(canvas_shape + (3, ))
    return image_rgb_2darray

freeze_shot(2.0) # compute single; for testing

# configure gif animation and save result to file
print('> Drawing animation. Please wait...')
t_grids = np.arange(0.5*np.pi, 3.5*np.pi, 0.05)
images_one_cycle = [freeze_shot(t) for t in t_grids]
images = []
for _ in range(1): images += images_one_cycle
imageio.mimsave('fourier_heart.gif', images, fps=10)
print('> Done!')
