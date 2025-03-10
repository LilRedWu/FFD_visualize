import numpy as np
from bernsetin import *
import utils
from utils import *
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
    # Normal Vectors
    tu = np.cross(t, u)
    su = np.cross(s, u)
    st = np.cross(s, t)

    diff = xyz - origin

    # TODO: vectorize? np.dot(diff, [tu, su, st]) / ...

    print("*****************")

    tu_divide = handling_inf(np.dot(diff, tu),np.dot(s, tu))
    su_divide = handling_inf(np.dot(diff, su),np.dot(t, su))
    st_divide = handling_inf(np.dot(diff, st),np.dot(u, st))

    stu = np.stack([
        tu_divide,
        su_divide,
        st_divide

    ], axis=-1)
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
    # get the contorol point in lattice space and convert it into origin coordinates
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

#
# def deform_mesh(xyz, lattice):
#     return trivariate_bernstein(lattice, xyz)


def get_stu_params(xyz):
    #setting up the necessary parameters to define a control lattice for Free-Form Deformation
    minimum, maximum = utils.extent(xyz, axis=0)
    stu_origin = minimum
    stu_axes = np.diag(maximum - minimum)
    stu_axes = maximum - minimum
    return stu_origin, stu_axes


def calculate_ffd(points, n=3):
    # import template_FFD.deform as ffd
    # import util3d.mesh.sample as sample
    # stu_origin, stu_axes = ffd.get_stu_params(vertices)

    norm_pointcloud = Normalize()(points)
    stu_origin, stu_axes = get_stu_params(norm_pointcloud)
    dims = (n,) * 3
    return get_ffd(norm_pointcloud, dims,stu_origin=stu_origin,stu_axes=stu_axes)