import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    images = np.array(images)
    N, h, w, c = images.shape

    lgt_t = lights.T
    A = lgt_t.dot(lights)
    albedo = np.zeros((h, w, c), dtype = np.float32)
    normals = np.zeros((h * w, 3), dtype = np.float32)
    inv_A = np.linalg.inv(A)

    images = images.reshape((N, h * w, c))
    for color in range(c):
        I_vt = images[:, :, color].swapaxes(1, 0)
        bs = np.array(list(map(lambda x: np.dot(lgt_t, x), I_vt)))
        xs = np.array(list(map(lambda b: np.dot(inv_A, b), bs)))
        albs = np.linalg.norm(xs, axis=1)
        norms = np.array(list(map(lambda x, alb: x / alb if alb >= 1e-7 else np.zeros(3), xs, albs)))
        normals += norms
        albedo[:, :, color] = albs.reshape((h, w))

    normals = normals.reshape((h, w, -1))
    normals /= c
    return albedo, normals


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    project_mat = np.dot(K, Rt)
    height, width = points.shape[:2]
    projections = np.zeros((height, width, 2))
    for i in range(height):
        for j in range(width):
            curr_pt = points[i, j]
            curr_pt = np.array([curr_pt[0], curr_pt[1], curr_pt[2], 1])
            proj_pt = np.dot(project_mat, curr_pt)
            projections[i, j] = proj_pt[:2] / proj_pt[2]

    return projections


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height, width, channels = image.shape
    half_patch = ncc_size // 2
    normalized = np.zeros((height, width, ncc_size, ncc_size, channels))
    for i in range(half_patch, height - half_patch):
        for j in range(half_patch, width - half_patch):
            normalized[i, j] = image[i - half_patch:i + half_patch + 1, j - half_patch:j + half_patch + 1]
    normalized = normalized.reshape((height, width, ncc_size ** 2, channels))
    normalized -= normalized.mean(axis=2, keepdims=True)
    normalized = normalized.reshape((height, width, -1), order='F').reshape((height * width, channels * ncc_size ** 2))
    norms = np.linalg.norm(normalized, axis=1)
    normalized = np.array(
        list(map(lambda x, y: x / y if y >= 1e-6 else np.zeros(channels * ncc_size ** 2), normalized, norms)))
    # temp = list(map(lambda x, y: x/y if y >= 1e-6 else np.zeros(channels * ncc_size ** 2), normalized.reshape((height*width, channels * ncc_size ** 2)), norms.reshape(height*width)))
    # temp = np.array(temp)
    normalized = normalized.reshape((height, width, channels * ncc_size ** 2))
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    vectors = image1 * image2
    ncc = np.sum(vectors, axis=2)
    return ncc
