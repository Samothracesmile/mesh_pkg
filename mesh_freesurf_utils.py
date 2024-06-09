from utils import *
from freesurfer_surface import Surface, Vertex, Triangle
from nibabel.freesurfer.io import read_morph_data
from mesh_utils import Mesh, merge_meshes, read_pobj
import nibabel.freesurfer.mghformat as fsmgh
import os

import numpy as np

import sys
sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/ydiffpak")
from diff_utils import *


def load_obj_vertices(lh_obj_surf_file, rh_obj_surf_file, resolution=1):
    '''Load lh and rh obj surface'''

    assert exists(lh_obj_surf_file)
    assert exists(rh_obj_surf_file)

    obj_lh_surf_vertices, lh_faces = read_pobj(lh_obj_surf_file)
    obj_rh_surf_vertices, rh_faces = read_pobj(rh_obj_surf_file)
    obj_lh_surf_vertices_imgspc = obj2imagespace(obj_lh_surf_vertices, resolution=resolution)
    obj_rh_surf_vertices_imgspc = obj2imagespace(obj_rh_surf_vertices, resolution=resolution)
    lh_obj_mesh, rh_obj_mesh = Mesh(obj_lh_surf_vertices_imgspc, lh_faces), Mesh(obj_rh_surf_vertices_imgspc, rh_faces)
    
    return lh_obj_mesh, rh_obj_mesh 



def freesurfer2obj_hcp(freesurfer_file, dpfix='ctx_obj', pfix='subspace'):
    freesurf2obj_hcp_Str = '/ifs/loni/faculty/shi/spectrum/yxia/yihao_code/mypackages/matlab/run_freesurf2obj_hcp.sh'
    '''convert freesurfer_file to obj_file'''
    obj_dir = dirname(dirname(freesurfer_file))
    obj_dir = pjoin(obj_dir, dpfix)
    create_dir(obj_dir)

    obj_file = basename(f'{freesurfer_file}.{pfix}.obj')
    obj_file = pjoin(obj_dir, obj_file)

    print('*')
    print(f'Converting {freesurfer_file} to {obj_file}')

    cmdStr = f'{freesurf2obj_hcp_Str} /usr/local/MATLAB/R2019b {freesurfer_file} {obj_file}'  
    job_name = '_'.join(freesurfer_file.split('/')[-5:])
    print(job_name.replace('/', ''))
    print(cmdStr)   
    os.system(cmdStr)
    # qsub.run7(job_name,cmdStr)

vol2fsa_func = '/ifs/loni/faculty/shi/spectrum/yxia/yihao_code/mypackages/ymedpak_v3/freesurfer_vol2fsa_surf.sh'
vol2sub_func = '/ifs/loni/faculty/shi/spectrum/yxia/yihao_code/mypackages/ymedpak_v3/freesurfer_vol2sub_surf.sh'
surf2surf_func = '/ifs/loni/faculty/shi/spectrum/yxia/yihao_code/mypackages/ymedpak_v3/freesurfer_surf2surf.sh'


def vol2fsa_surf(projdir, subjname, subvolname, outprj_fsa_lh, outprj_fsa_rh, surfname='pial'):
    
    '''
    projdir: directory of project
    subjname: name of subject
    subvolname: filename of vol data to project (3D or 4D) nifty
    surfname: 'pial' or 'white'
    outprj_fsa_lh, outprj_fsa_rh: the function on the fsaverage surface for lh and rh
    '''
    
    ex_commond = ['bash', vol2fsa_func, projdir, subjname, 
        subvolname, surfname, outprj_fsa_lh, outprj_fsa_rh]
    ex_commond = ' '.join(ex_commond)
    # print(ex_commond)
    os.system(ex_commond) 
    

def vol2sub_surf(projdir, subjname, subvolname, outprj_lh, outprj_rh, surfname='pial'):
    
    '''
    projdir: directory of project
    subjname: name of subject
    subvolname: filename of vol data to project (3D or 4D) nifty
    surfname: 'pial' or 'white'
    outprj_lh, outprj_rh: the function on the subject surface for lh and rh
    '''
    
    ex_commond = ['bash', vol2sub_func, projdir, subjname, 
        subvolname, surfname, outprj_lh, outprj_rh]
    ex_commond = ' '.join(ex_commond)
    # print(ex_commond)
    os.system(ex_commond) 


def surf2surf(projdir, srcsubject, trgsubject, 
            srcval_lh, srcval_rh, trgval_lh, trgval_rh, force_flg=False):
    
    '''
    Transform surf val from srcsubject to trgsubject 
        fsaverage can be used for transfrom inbetween common space and subject space

    projdir: directory of project

    '''
    if not (exists(trgval_lh) and exists(trgval_rh)) or force_flg:
    
        ex_commond = ['bash', surf2surf_func, projdir, srcsubject, 
            trgsubject, srcval_lh, srcval_rh, trgval_lh, trgval_rh]
        ex_commond = ' '.join(ex_commond)
        # print(ex_commond)
        os.system(ex_commond) 


def fsa2sub_surf(dmri_ctx_data, scaler_lh, scaler_rh, sub_scaler_lh, sub_scaler_rh):
    surf2surf(dmri_ctx_data.ctx_proj_dir, 'fsaverage', dmri_ctx_data.sub_name, 
            scaler_lh, scaler_rh, sub_scaler_lh, sub_scaler_rh)
    
def sub2fsa_surf(dmri_ctx_data, scaler_lh, scaler_rh, fsa_scaler_lh, fsa_scaler_rh):
    surf2surf(dmri_ctx_data.ctx_proj_dir, dmri_ctx_data.sub_name, 'fsaverage',
            scaler_lh, scaler_rh, fsa_scaler_lh, fsa_scaler_rh)


def load_mgh_scalers(mgh_file):
    '''
    load the morph data from .mgh file
    Input:
        mgh filename (string)
    Output:
        vertices attribute np.array
    '''
    
    mgh = fsmgh.load(mgh_file)
    return np.squeeze(mgh.get_fdata())
    

def save_mgh_scalers(data, pro_mgh_file, save_mgh_file):
    pro_mgh = fsmgh.load(pro_mgh_file)
    data = np.expand_dims(data, axis=(1,2))
    create_dir(dirname(save_mgh_file))
    save_mgh = fsmgh.MGHImage(data, pro_mgh.affine, 
             header=pro_mgh.header, extra=pro_mgh.extra, file_map=pro_mgh.file_map)
    
    fsmgh.save(save_mgh, save_mgh_file)


def freesurf2imagespace_gen(vertices, resolution, image_shape):
    
    '''convert the freesurfer mesh vertices to image space'''
    
    return vertices/resolution + image_shape/2


def obj2imagespace_gen(vertices, resolution, image_shape, shift_array=None):
    
    '''convert the pobj mesh vertices to image space'''
    if shift_array is None:
        vertices = vertices - np.array([90.3, 108.5, 90])
    else:
        vertices = vertices - shift_array

    return vertices/resolution + image_shape/2

def freesurf2imagespace(vertices, resolution=1, shift_array=np.array([90.3, 108.5, 90])):

    '''convert the freesurfer mesh vertices to image space'''
    # shift_array=np.array([89, 108.5, 91])
    return (vertices + shift_array)/resolution

def obj2imagespace(vertices, resolution=1):
    '''convert the pobj mesh vertices to image space'''
    return vertices/resolution

def obj2freesurf(vertices, shift_array=np.array([90.3, 108.5, 90])):

    return vertices - shift_array

import nibabel as nib
def load_freesurfer_n(freesurfer_file):
    vertices, faces, create_stamp, volume_info = nib.freesurfer.io.read_geometry(freesurfer_file, read_metadata=True, read_stamp=True)
    return vertices, faces, create_stamp, volume_info

def save_freesurfer_n(freesurfer_file, vertices, faces, volume_info, create_stamp):
    create_dir(dirname(freesurfer_file))
    nib.freesurfer.io.write_geometry(freesurfer_file, vertices, faces, create_stamp=create_stamp, volume_info=volume_info)

def load_freesurfer_annot_n(annot_file):
    label, ctab, names = nib.freesurfer.io.read_annot(annot_file)
    return label, ctab, names


def load_freesurfer_annot(annot_file):
    label, ctab, names = nib.freesurfer.io.read_annot(annot_file)
    return label, ctab, names

def load_freesurfer(freesurfer_file):
    freesurfer_surf = Surface.read_triangular(freesurfer_file)
    
    vertices = np.vstack([np.array([vertic.right,vertic.anterior,vertic.superior]) for vertic in freesurfer_surf.vertices])
    faces = np.vstack([np.array(face.vertex_indices) for face in freesurfer_surf.triangles])
    
    return vertices, faces

def save_freesurfer(freesurfer_file, vertices, faces, ref_freesurf_file):

    assert exists(ref_freesurf_file)
    ref_freesurf_surf = Surface.read_triangular(ref_freesurf_file)

    freesurfer_surf = Surface()
    for vert in vertices:
        freesurfer_surf.add_vertex(Vertex(*vert))
        
    for face in faces:
        freesurfer_surf.triangles.append(Triangle(tuple(int(a) for a in face)))  

    freesurfer_surf.volume_geometry_info = ref_freesurf_surf.volume_geometry_info
    freesurfer_surf.creator = ref_freesurf_surf.creator
    freesurfer_surf.creation_datetime = ref_freesurf_surf.creation_datetime
    freesurfer_surf.using_old_real_ras = ref_freesurf_surf.using_old_real_ras
    freesurfer_surf.command_lines = ref_freesurf_surf.command_lines
    freesurfer_surf.annotation = ref_freesurf_surf.annotation

    freesurfer_surf.write_triangular(freesurfer_file)


def gen_mids_surface(p_freesurfer_file, w_freesurfer_file, m_freesurfer_file):
    
    '''
    Generate middle surface between pial and white using freesurfer py
    
    p_freesurfer_file: freesurfer pial surface file
    w_freesurfer_file: freesurfer white surface file 
    
    m_freesurfer_file: save mids surface file
    '''
    p_vertices, p_faces = load_freesurfer(p_freesurfer_file)
    w_vertices, w_faces = load_freesurfer(w_freesurfer_file)
    
    assert np.sum(np.abs(p_faces-w_faces)) < 0.01, f'Triangler Mesh of {p_freesurfer_file} and {p_freesurfer_file} does not match!'

    # generate mids surface
    m_vertices = np.mean(np.array([p_vertices, w_vertices]), axis=0)
    save_freesurfer(m_freesurfer_file, m_vertices, p_faces, p_freesurfer_file)
    

    
def gen_mids_surface_n(p_freesurfer_file, w_freesurfer_file, m_freesurfer_file):
    
    '''
    Generate middle surface between pial and white using nibable
    
    p_freesurfer_file: freesurfer pial surface file
    w_freesurfer_file: freesurfer white surface file 
    
    m_freesurfer_file: save mids surface file
    '''
    p_vertices, p_faces, p_create_stamp, p_volume_info = load_freesurfer_n(p_freesurfer_file)
    w_vertices, w_faces, w_create_stamp, w_volume_info = load_freesurfer_n(w_freesurfer_file)

    assert np.sum(np.abs(p_faces-w_faces)) < 0.01, f'Triangler Mesh of {p_freesurfer_file} and {p_freesurfer_file} does not match!'
    
    # generate mids surface
    m_vertices = np.mean(np.array([p_vertices,w_vertices]), axis=0)
    m_faces, m_create_stamp, m_volume_info = p_faces, p_create_stamp, p_volume_info

    save_freesurfer_n(m_freesurfer_file, m_vertices, m_faces, m_create_stamp, m_volume_info)



def freesurfer2mesh(freesurfer_file, mgh_scaler_file=None):
    scalers=None
    if mgh_scaler_file:
        scalers = load_mgh_scalers(mgh_scaler_file)
    
    vertices, faces = load_freesurfer(freesurfer_file)
    
    return Mesh(vertices, faces, scalers)

def surf_stat(surf_list):
    surf_stack = np.stack([load_mgh_scalers(mgh) for mgh in surf_list])
    surf_std = np.std(surf_stack,axis=0)
    surf_mean = np.mean(surf_stack,axis=0)
    surf_cov = surf_std/surf_mean
    
    if np.sum(surf_mean == 0) > 0:
        print(f'surf_mean contains 0')
        
    return surf_stack, surf_std, surf_mean, surf_cov

# def extr_common_postfix(string1, string2):
#     from difflib import SequenceMatcher
#     match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))

#     return string1[match.a: match.a + match.size]

# def common_postfix(string_list):
#     string_list = [basename(file) for file in string_list]
    
#     common_string = string_list[0]
#     for string in string_list[1:]:
#         common_string = extr_common_postfix(common_string, string)
        
#     return common_string

# def create_dir(_dir):
#     if not exists(_dir):
#         makedirs(_dir)     
        
        
def gen_scaler_img(ref_surf_list, tar_surf_list, scaler_filename=None, pro_mgh_file=None, epsi=0.001, thr=5):
    
    _, _, ref_surf_mean, _ = surf_stat(ref_surf_list)
    _, _, tar_surf_mean, _ = surf_stat(tar_surf_list)

    scaler = np.sqrt(ref_surf_mean/(tar_surf_mean+epsi))
    
    if thr > 0:
        scaler[scaler > thr] = thr
    
    if not pro_mgh_file:
        pro_mgh_file = ref_surf_list[0]
    
    if scaler_filename:
        save_mgh_scalers(scaler, pro_mgh_file, scaler_filename)
    
    return scaler


def mgh2npy(mgh_file, npy_file):
    
    scalars = load_mgh_scalers(mgh_file)
    np.save(npy_file, scalars)

    return scalars


import sklearn
from scipy import sparse, interpolate

def face_outer_normals(mesh):
    """Get the normal to each triangle in a mesh.
    They are the outer normals if the mesh respects the convention that the
    direction given by the direct order of a triangle's vertices (right-hand
    rule) points outwards.
    """
    vertices, faces = load_surf_mesh(mesh)
    face_vertices = vertices[faces]
    # The right-hand rule gives the direction of the outer normal
    normals = np.cross(face_vertices[:, 1, :] - face_vertices[:, 0, :],
                       face_vertices[:, 2, :] - face_vertices[:, 0, :])
    normals = sklearn.preprocessing.normalize(normals)
    return normals


def surrounding_faces(mesh):
    """Get matrix indicating which faces the nodes belong to.
    i, j is set if node i is a vertex of triangle j.
    """
    vertices, faces = load_surf_mesh(mesh)
    n_faces = faces.shape[0]
    return sparse.csr_matrix((np.ones(3 * n_faces), (faces.ravel(), np.tile(
        np.arange(n_faces), (3, 1)).T.ravel())), (vertices.shape[0], n_faces))


def vertex_outer_normals(mesh):
    """Get the normal at each vertex in a triangular mesh.
    They are the outer normals if the mesh respects the convention that the
    direction given by the direct order of a triangle's vertices (right-hand
    rule) points outwards.
    """
    vertex_faces = surrounding_faces(mesh)
    face_normals = face_outer_normals(mesh)
    normals = vertex_faces.dot(face_normals)
    return sklearn.preprocessing.normalize(normals)


def vol_2_surface(img_file, tar_mesh, offset_half_width, n_points, interpolation_method):
    '''
    Project volume data onto target mesh
    '''

    # A. Sample Mesh LLocations
    vertices = tar_mesh.vertices
    faces = tar_mesh.faces

    if offset_half_width > 0.5:
        offsets = np.linspace(offset_half_width, -offset_half_width, n_points)
        normals = vertex_outer_normals(tar_mesh)
        sample_locations = vertices[np.newaxis, :, :] + normals * offsets[:, np.newaxis, np.newaxis]
    else:
        sample_locations = vertices[np.newaxis, :, :]

    sample_locations = np.rollaxis(sample_locations, 1)
    n_vertices, n_points, img_dim = sample_locations.shape

    # B. Load image and get grid
    img_data = load(img_file).get_fdata()

    # unify to 4d
    if len(img_data.shape) == 4:
        img_data = np.rollaxis(img_data, -1)
    else:
        img_data = img_data[np.newaxis, ...]

    grid = [np.arange(size) for size in img_data[0].shape]

    # C. interpolation
    interp_locations = np.vstack(sample_locations)
    all_samples = []
    for img in img_data:
        interpolator = interpolate.RegularGridInterpolator(grid, img,
            bounds_error=False, method=interpolation_method, fill_value=None)
        samples = interpolator(interp_locations)
        all_samples.append(samples)
    all_samples = np.asarray(all_samples)
    all_samples = all_samples.reshape((len(img_data), n_vertices, n_points))
    
    return sample_locations, all_samples