import numpy as np
import utils
from bernstein import bernstein_poly, trivariate_bernstein
from point_utils import *

def handling_inf(A,B):
    with np.errstate(divide='ignore', invalid='ignore'):
        C = A / B
        # 可以参考的对错误处理的方式
        #     arrayC[arrayC == np.inf] = 0  # 对inf的错误进行修正，不会修正-inf
        C[~ np.isfinite(C)] = 0  # 对 -inf, inf, NaN进行修正，置为0

        return C
    


def xyz_to_stu(xyz, origin, stu_axes):
    if stu_axes.shape == (3,):
        stu_axes = np.diag(stu_axes)
        # raise ValueError(
        #     'stu_axes should have shape (3,), got %s' % str(stu_axes.shape))
    # s, t, u = np.diag(stu_axes)
    assert(stu_axes.shape == (3, 3))
    s, t, u = stu_axes

    tu = np.cross(t, u)
    su = np.cross(s, u)
    st = np.cross(s, t)

    diff = xyz - origin

    # TODO: vectorize? np.dot(diff, [tu, su, st]) / ...

    try:
        stu = np.stack([
            np.dot(diff, tu) / np.dot(s, tu),
            np.dot(diff, su) / np.dot(t, su),
            np.dot(diff, st) / np.dot(u, st)
        ], axis=-1)
    except RuntimeWarning:
        print('okk')


    return stu


def stu_to_xyz(stu_points, stu_origin, stu_axes):
    if stu_axes.shape != (3,):
        raise NotImplementedError()
    return stu_origin + stu_points*stu_axes


def get_stu_control_points(dims):
    stu_lattice = utils.mesh3d(
        *(np.linspace(0, 1, d+1) for d in dims), dtype=np.float32)
    stu_points = np.reshape(stu_lattice, (-1, 3))
    return stu_points


def get_control_points(dims, stu_origin, stu_axes):
    stu_points = get_stu_control_points(dims)
    xyz_points = stu_to_xyz(stu_points, stu_origin, stu_axes)
    return xyz_points


def get_stu_deformation_matrix(stu, dims):
    v = utils.mesh3d(
        *(np.arange(0, d+1, dtype=np.int32) for d in dims),
        dtype=np.int32)
    v = np.reshape(v, (-1, 3))

    weights = bernstein_poly(
        n=np.array(dims, dtype=np.int32),
        v=v,
        stu=np.expand_dims(stu, axis=-2))

    b = np.prod(weights, axis=-1)
    return b


def get_deformation_matrix(xyz, dims, stu_origin, stu_axes):
    stu = xyz_to_stu(xyz, stu_origin, stu_axes)
    return get_stu_deformation_matrix(stu, dims)


def get_ffd(xyz, dims, stu_origin=None, stu_axes=None):
    if stu_origin is None or stu_axes is None:
        if not (stu_origin is None and stu_axes is None):
            raise ValueError(
                'Either both or neither of stu_origin/stu_axes must be None')
        stu_origin, stu_axes = get_stu_params(xyz)
    b = get_deformation_matrix(xyz, dims, stu_origin, stu_axes)
    p = get_control_points(dims, stu_origin, stu_axes)
    return b, p, xyz


def deform_mesh(xyz, lattice):
    return trivariate_bernstein(lattice, xyz)


def get_stu_params(xyz):
    minimum, maximum = utils.extent(xyz, axis=0)
    stu_origin = minimum
    # stu_axes = np.diag(maximum - minimum)
    stu_axes = maximum - minimum
    return stu_origin, stu_axes

def _calculate_ffd(vertices, faces, n=3, n_samples=None):
    # import template_FFD.deform as ffd
    # import util3d.mesh.sample as sample
    # stu_origin, stu_axes = ffd.get_stu_params(vertices)

    if n_samples is None:
        points =  PointSampler(3000)((vertices, faces))
        norm_pointcloud = Normalize()(points)
        stu_origin, stu_axes = get_stu_params(norm_pointcloud)

    else:
        pointcloud = PointSampler()((vertices, faces))
    dims = (n,) * 3
    # return ffd.get_ffd(points, dims)
    return get_ffd(norm_pointcloud, dims,stu_origin=stu_origin,stu_axes=stu_axes)