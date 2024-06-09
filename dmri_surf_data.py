import os 

from collections import namedtuple
from mesh_freesurf_utils import vol2fsa_surf, vol2sub_surf, surf2surf, obj2imagespace, load_obj_vertices, vol_2_surface, load_mgh_scalers, save_mgh_scalers
from mesh_utils import Mesh, merge_meshes, read_pobj

from nilearn.surface import load_surf_mesh

import sys
sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/ydiffpak")
from diff_utils import *

# Mesh = namedtuple("mesh", ["coordinates", "faces"])
# Surface = namedtuple("surface", ["mesh", "data"])

class freesurf_data():
    '''
    freesurfer data class
    proj_dir: the project directory of surface data
    sub_name: the subject name of the freesurfer data
    surfname: white or pial
    '''

    def __init__(self, proj_dir, sub_name, surfname='pial', annot_pfix='aparc.annot'):

        self.proj_dir = proj_dir
        self.sub_name = sub_name
        self.surfname = surfname

        self.lh_surf_file = os.path.join(proj_dir, sub_name, 'surf', f'lh.{surfname}')
        self.rh_surf_file = os.path.join(proj_dir, sub_name, 'surf', f'rh.{surfname}')

        self.lh_annot_file = os.path.join(proj_dir, sub_name, 'label', f'lh.{annot_pfix}')  #.aparc.a2009s.annot, #.aparc.DKTatlas40.annot
        self.rh_annot_file = os.path.join(proj_dir, sub_name, 'label', f'rh.{annot_pfix}')

        self.lh_thick_file = os.path.join(proj_dir, sub_name, 'surf', f'lh.thickness')  #.aparc.a2009s.annot, #.aparc.DKTatlas40.annot
        self.rh_thick_file = os.path.join(proj_dir, sub_name, 'surf', f'rh.thickness')

    def add_obj_surf_file(self, lh_obj_surf_file, rh_obj_surf_file, obj_spacename=''):

        assert os.path.exists(lh_obj_surf_file)
        assert os.path.exists(rh_obj_surf_file)

        self.lh_obj_surf_file = lh_obj_surf_file
        self.rh_obj_surf_file = rh_obj_surf_file
        self.obj_spacename = obj_spacename

    # def load_mesh(self, resolution=1, shift_array=np.array([90.3, 108.5, 90])):

    #     lh_surf_vertices, lh_faces = load_freesurfer(self.lh_surf_file)
    #     rh_surf_vertices, rh_faces = load_freesurfer(self.rh_surf_file)
    #     lh_surf_vertices_imgspc = freesurf2imagespace(lh_surf_vertices, resolution=resolution, shift_array=shift_array)
    #     rh_surf_vertices_imgspc = freesurf2imagespace(rh_surf_vertices, resolution=resolution, shift_array=shift_array)

    #     return Mesh(lh_surf_vertices_imgspc, lh_faces), Mesh(rh_surf_vertices_imgspc, rh_faces)

    def load_mesh_nilearn(self, affine=None):
        lh_mesh_nilearn = load_surf_mesh(self.lh_surf_file)
        rh_mesh_nilearn = load_surf_mesh(self.rh_surf_file)

        if affine is not None:

            lh_mesh_nilearn_vertices_imgspc = np.array(resampling.coord_transform(*lh_mesh_nilearn.coordinates.T, affine=np.linalg.inv(affine))).T
            rh_mesh_nilearn_vertices_imgspc = np.array(resampling.coord_transform(*rh_mesh_nilearn.coordinates.T, affine=np.linalg.inv(affine))).T

            return Mesh(lh_mesh_nilearn_vertices_imgspc, lh_mesh_nilearn.faces), Mesh(rh_mesh_nilearn_vertices_imgspc, rh_mesh_nilearn.faces)

        else:
            return Mesh(lh_mesh_nilearn.coordinates, lh_mesh_nilearn.faces), Mesh(rh_mesh_nilearn.coordinates, rh_mesh_nilearn.faces)


    # def load_obj_mesh(self, resolution=1):

    #     assert os.path.exists(self.lh_obj_surf_file)
    #     assert os.path.exists(self.rh_obj_surf_file)

    #     lh_obj_mesh, rh_obj_mesh = load_obj_vertices(self.lh_obj_surf_file, self.rh_obj_surf_file, resolution=resolution)

    #     return lh_obj_mesh, rh_obj_mesh


    def simple_vol2surf(self, dmri_data, obj_spacename='dti', interpolation_method='linear', offset_half_width=0.01, n_points=1):

        # Using naive freesurfer projection
        free_prj_lh = dmri_data.dwi_file.replace('.nii.gz', f'.lh.{self.surfname}.mgh')
        free_prj_rh = dmri_data.dwi_file.replace('.nii.gz', f'.rh.{self.surfname}.mgh')
        if not (os.path.exists(free_prj_lh) and os.path.exists(free_prj_rh)): 
            self.prj_vol2sub_surf(dmri_data.dwi_file, free_prj_lh, free_prj_rh)

        # Using interpolation for projection
        dwi_nii = load(dmri_data.dwi_file)
        resolution = np.abs(dwi_nii.affine[0,0])

        inter_prj_lh = dmri_data.dwi_file.replace('.nii.gz', f'.lh.{self.surfname}.{obj_spacename}.{interpolation_method}.mgh')
        inter_prj_rh = dmri_data.dwi_file.replace('.nii.gz', f'.rh.{self.surfname}.{obj_spacename}.{interpolation_method}.mgh')

        print(free_prj_lh, free_prj_rh)
        print(inter_prj_lh, inter_prj_rh)

        self.free_prj_lh = free_prj_lh
        self.free_prj_rh = free_prj_rh


        if not (os.path.exists(inter_prj_lh) and os.path.exists(inter_prj_rh)): 

            self.lh_obj_mesh, self.rh_obj_mesh = load_obj_vertices(self.lh_obj_surf_file, self.rh_obj_surf_file, resolution=resolution)

            _, all_samples = vol_2_surface(dmri_data.dwi_file, self.lh_obj_mesh, offset_half_width, n_points, interpolation_method)
            lh_dwi_val = np.squeeze(all_samples).T

            _, all_samples = vol_2_surface(dmri_data.dwi_file, self.rh_obj_mesh, offset_half_width, n_points, interpolation_method)
            rh_dwi_val = np.squeeze(all_samples).T

            save_mgh_scalers(lh_dwi_val, free_prj_lh, inter_prj_lh)
            save_mgh_scalers(rh_dwi_val, free_prj_rh, inter_prj_rh)

        self.inter_prj_lh = inter_prj_lh
        self.inter_prj_rh = inter_prj_rh

        # return (free_prj_lh, free_prj_rh), (inter_prj_lh, inter_prj_rh)


    def prj_vol2sub_surf(self, subvolname, outprj_sub_lh, outprj_sub_rh, force_flg=False):
        '''
        project volume data onto subject surface (using freesurfer)
        '''
        if not (os.path.exists(outprj_sub_rh) and os.path.exists(outprj_sub_lh)) or force_flg:
            vol2sub_surf(self.proj_dir, self.sub_name, subvolname, 
                outprj_sub_lh, outprj_sub_rh, surfname=self.surfname)
        else:
            print(f'Using existed {outprj_sub_rh} and {outprj_sub_lh}')


    def prj_vol2fsa_surf(self, subvolname, outprj_fsa_lh, outprj_fsa_rh, force_flg=False):
        '''
        project volume data onto fsaverage surface (using freesurfer)
        '''
        if not (os.path.exists(outprj_fsa_rh) and os.path.exists(outprj_fsa_lh)) or force_flg:
            # print('res')
            vol2fsa_surf(self.proj_dir, self.sub_name, subvolname, 
                outprj_fsa_lh, outprj_fsa_rh, surfname=self.surfname)
        else:
            print(f'Using existed {outprj_fsa_rh} and {outprj_fsa_lh}')


 # class objsurf_data():
 #    '''
 #    freesurfer data class
 #    proj_dir: the project directory of surface data
 #    sub_name: the subject name of the freesurfer data
 #    surfname: white or pial
 #    '''

 #    def __init__(self, proj_dir, sub_name, surfname='pial'):
 #        self.proj_dir = proj_dir
 #        self.sub_name = sub_name
 #        self.surfname = surfname

 #        self.lh_surf_file = os.path.join(proj_dir, sub_name, 'surf', f'lh.{surfname}')
 #        self.rh_surf_file = os.path.join(proj_dir, sub_name, 'surf', f'rh.{surfname}')

 #        self.lh_annot_file = os.path.join(proj_dir, sub_name, 'label', f'lh.aparc.annot')  #.aparc.a2009s.annot, #.aparc.DKTatlas40.annot
 #        self.rh_annot_file = os.path.join(proj_dir, sub_name, 'label', f'rh.aparc.annot')

# class surface_data():
#     '''
#     freesurfer data class
#     proj_dir: the project directory of surface data
#     sub_name: the subject name of the freesurfer data
#     surfname: white or pial
#     '''

#     def __init__(self, proj_dir, sub_name, surfname='pial'):
#         self.proj_dir = proj_dir
#         self.sub_name = sub_name
#         self.surfname = surfname


#     def prj_vol2fsa_surf(self, subvolname, outprj_fsa_lh, outprj_fsa_rh, force_flg=False):
#         '''
#         project volume data onto fsaverage surface (using freesurfer)
#         '''
#         if not (os.path.exists(outprj_fsa_rh) and os.path.exists(outprj_fsa_lh)) or force_flg:
#             # print('res')
#             vol2fsa_surf(self.proj_dir, self.sub_name, subvolname, 
#                 outprj_fsa_lh, outprj_fsa_rh, surfname=self.surfname)
#         else:
#             print(f'Using existed {outprj_fsa_rh} and {outprj_fsa_lh}')


#     def prj_vol2sub_surf(self, subvolname, outprj_fsa_lh, outprj_fsa_rh, force_flg=False):
#         '''
#         project volume data onto subject surface (using freesurfer)
#         '''
#         if not (os.path.exists(outprj_fsa_rh) and os.path.exists(outprj_fsa_lh)) or force_flg:
#             vol2sub_surf(self.proj_dir, self.sub_name, subvolname, 
#                 outprj_fsa_lh, outprj_fsa_rh, surfname=self.surfname)
#         else:
#             print(f'Using existed {outprj_fsa_rh} and {outprj_fsa_lh}')


from dmri_ctx_feature import gen_cdti, gen_crish, gen_cmoment

class hemi_diffusion_data():
    '''
    dMRI ctx data
    '''
    
    def __init__(self, dwi_file, bvec_file, bval_file, sub_name, ctx_pfix, sub_dir, hemi_name):
        
        # file info.
        self.dwi_file = dwi_file
        self.bvec_file = bvec_file
        self.bval_file = bval_file
        
        # subject info.
        self.sub_name = sub_name
        self.sub_dir = sub_dir

        self.hemi_name = hemi_name
        self.ctx_pfix = ctx_pfix

    def make_dti(self, dti_dir='DTI', b0_thr=100, tar_bval=None, force_flg=False, assert_flg=False):
        
        # make dti_dir
        sub_dti_dir = os.path.join(self.sub_dir, dti_dir)
        create_dir(sub_dti_dir)
        
        # DTI filename
        self.fa_file = os.path.join(sub_dti_dir, f'{self.sub_name}.FA{self.ctx_pfix}')
        self.md_file = os.path.join(sub_dti_dir, f'{self.sub_name}.MD{self.ctx_pfix}')
        self.gfa_file = os.path.join(sub_dti_dir, f'{self.sub_name}.GFA{self.ctx_pfix}')

        if assert_flg:
            assert os.path.exists(self.fa_file), f'{self.fa_file} should os.path.exists in evaluation stage!'
            assert os.path.exists(self.md_file), f'{self.md_file} should os.path.exists in evaluation stage!'
            assert os.path.exists(self.gfa_file), f'{self.gfa_file} should os.path.exists in evaluation stage!'

        # calculate DTI
        gen_cdti(self.dwi_file, self.bval_file, self.bvec_file, 
            self.fa_file, self.md_file, self.gfa_file, 
            b0_thr=b0_thr, tar_bval=tar_bval, force_flg=force_flg)

  
    def make_rish(self, rish_dir='RISH', sh_num=8, b0_thr=100, force_flg=False):
        
        # make sub_rish_dir
        sub_rish_dir = os.path.join(self.sub_dir, rish_dir)
        create_dir(sub_rish_dir)
        
        # RISH filename pattern
        rish_file_pattern = os.path.join(sub_rish_dir, f'{self.sub_name}_rish_l*{self.ctx_pfix}')

        # calculate RISH 
        self.shs_files = gen_crish(self.dwi_file, self.bval_file, self.bvec_file,
                                    rish_file_pattern, sh_num, b0_thr=b0_thr, force_flg=force_flg)
        
        
    def make_moment(self, moment_dir='Moment', b0_thr=100, force_flg=False):
    
        # make sub_rish_dir
        sub_moment_dir = os.path.join(self.sub_dir, moment_dir)
        create_dir(sub_moment_dir)
        
        # Moment filename pattern
        moment_file_pattern = os.path.join(sub_moment_dir, f'{self.sub_name}_moment*{self.ctx_pfix}')

        self.moment_files = gen_cmoment(self.dwi_file, self.bval_file, self.bvec_file,
                            moment_file_pattern, b0_thr=b0_thr, force_flg=force_flg)
    

class ctx_diffusion_data():
    def __init__(self, lh_dwi_mgh, rh_dwi_mgh, sub_name, data_pfix, surf_name=None, space_name=None, ctx_proj_dir=None):
    # def __init__(self, lh_dwi_mgh, rh_dwi_mgh, dmri_data, data_pfix, surf_name=None, space_name=None, ctx_proj_dir=None):

        assert os.path.exists(lh_dwi_mgh), f'{lh_dwi_mgh} does not os.path.exists!'
        assert os.path.exists(rh_dwi_mgh), f'{rh_dwi_mgh} does not os.path.exists!'

        sub_dir = dirname(lh_dwi_mgh)
        sub_dir1 = dirname(rh_dwi_mgh)

        assert sub_dir == sub_dir1

        self.sub_name = sub_name
        # self.sub_name = dmri_data.sub_name
        self.ctx_proj_dir = ctx_proj_dir
        self.surf_name = surf_name
        self.space_name = space_name

        self.bvec_file = os.path.join(sub_dir, f'{data_pfix}.bvec')
        self.bval_file = os.path.join(sub_dir, f'{data_pfix}.bval')

        assert self.bvec_file == os.path.join(dirname(rh_dwi_mgh), f'{data_pfix}.bvec')
        assert self.bval_file == os.path.join(dirname(rh_dwi_mgh), f'{data_pfix}.bval')

        lh_ctx_pfix = basename(lh_dwi_mgh).replace(data_pfix, '')
        self.lh = hemi_diffusion_data(lh_dwi_mgh, self.bvec_file, self.bval_file, self.sub_name, 
                                    lh_ctx_pfix, sub_dir, hemi_name='lh')
        
        rh_ctx_pfix = basename(rh_dwi_mgh).replace(data_pfix, '')
        self.rh = hemi_diffusion_data(rh_dwi_mgh, self.bvec_file, self.bval_file, self.sub_name, 
                                    rh_ctx_pfix, sub_dir, hemi_name='rh')


        # lh_ctx_pfix = basename(lh_dwi_mgh).replace(data_pfix, '')
        # self.lh = hemi_diffusion_data(lh_dwi_mgh, dmri_data.bvec_file, dmri_data.bval_file, self.sub_name, 
        #                             lh_ctx_pfix, sub_dir, hemi_name='lh')
        
        # rh_ctx_pfix = basename(rh_dwi_mgh).replace(data_pfix, '')
        # self.rh = hemi_diffusion_data(rh_dwi_mgh, dmri_data.bvec_file, dmri_data.bval_file, self.sub_name, 
        #                             rh_ctx_pfix, sub_dir, hemi_name='rh')


    def gen_dti_features(self, data_pfix='data_pfix', tar_bval=None, assert_flg=False):
    
        # generate the dti data
        # data_name = '.'.join(basename(self.lh.dwi_file).split('.')[:-3])
        if tar_bval is not None:
            self.lh.make_dti(f'{data_pfix}_{tar_bval}_ctx_dti', tar_bval=tar_bval, assert_flg=assert_flg)
            self.rh.make_dti(f'{data_pfix}_{tar_bval}_ctx_dti', tar_bval=tar_bval, assert_flg=assert_flg)
        else:
            print(tar_bval)
            self.lh.make_dti(f'{data_pfix}_ctx_dti', tar_bval=tar_bval, assert_flg=assert_flg)
            self.rh.make_dti(f'{data_pfix}_ctx_dti', tar_bval=tar_bval, assert_flg=assert_flg)
            
        self.lh.fa_fsa_file = self.lh.fa_file.replace('.lh', '_fsa.lh')
        self.rh.fa_fsa_file = self.rh.fa_file.replace('.rh', '_fsa.rh')
        surf2surf(self.ctx_proj_dir, self.sub_name, 'fsaverage',
                    self.lh.fa_file, self.rh.fa_file, self.lh.fa_fsa_file, self.rh.fa_fsa_file)
        
        self.lh.gfa_fsa_file = self.lh.gfa_file.replace('.lh', '_fsa.lh')
        self.rh.gfa_fsa_file = self.rh.gfa_file.replace('.rh', '_fsa.rh')
        surf2surf(self.ctx_proj_dir, self.sub_name, 'fsaverage',
                    self.lh.gfa_file, self.rh.gfa_file, self.lh.gfa_fsa_file, self.rh.gfa_fsa_file)
        
        self.lh.md_fsa_file = self.lh.md_file.replace('.lh', '_fsa.lh')
        self.rh.md_fsa_file = self.rh.md_file.replace('.rh', '_fsa.rh')
        surf2surf(self.ctx_proj_dir, self.sub_name, 'fsaverage',
                    self.lh.md_file, self.rh.md_file, self.lh.md_fsa_file, self.rh.md_fsa_file)


    def gen_rish_features(self, data_pfix='data_pfix'):
        # generate rish in subject space
        self.lh.make_rish(rish_dir=f'{data_pfix}_ctx_rish')
        self.rh.make_rish(rish_dir=f'{data_pfix}_ctx_rish')

        # mapping rish to common space
        self.lh.shs_fsa_files = []
        self.rh.shs_fsa_files = []
        for lh_shs_file, rh_shs_file in zip(self.lh.shs_files, self.rh.shs_files):

            lh_shs_fsa_file = lh_shs_file.replace('.lh', '_fsa.lh')
            rh_shs_fsa_file = rh_shs_file.replace('.rh', '_fsa.rh')
            
            if not (os.path.exists(lh_shs_fsa_file) and os.path.exists(rh_shs_fsa_file)):
                surf2surf(self.ctx_proj_dir, self.sub_name, 'fsaverage',
                                lh_shs_file, rh_shs_file, lh_shs_fsa_file, rh_shs_fsa_file)

            self.lh.shs_fsa_files.append(lh_shs_fsa_file)
            self.rh.shs_fsa_files.append(rh_shs_fsa_file)


    def gen_moment_features(self, data_pfix='data_pfix'):
        # generate moment in subject space
        self.lh.make_moment(f'{data_pfix}_ctx_mom')
        self.rh.make_moment(f'{data_pfix}_ctx_mom')

        # mapping moment to common space
        self.lh.moment_fsa_files = []
        self.rh.moment_fsa_files = []
        for lh_moment_file, rh_moment_file in zip(self.lh.moment_files, self.rh.moment_files):

            lh_moment_fsa_file = lh_moment_file.replace('.lh', '_fsa.lh')
            rh_moment_fsa_file = rh_moment_file.replace('.rh', '_fsa.rh')
            
            if not (os.path.exists(lh_moment_fsa_file) and os.path.exists(rh_moment_fsa_file)):
                surf2surf(self.ctx_proj_dir, self.sub_name, 'fsaverage',
                                lh_moment_file, rh_moment_file, lh_moment_fsa_file, rh_moment_fsa_file)

            self.lh.moment_fsa_files.append(lh_moment_fsa_file)
            self.rh.moment_fsa_files.append(rh_moment_fsa_file)


    def sub2fsa(self, lh_sub_file, rh_sub_file, lh_fsa_file, rh_fsa_file):
        '''Warp subspace surface to fsaverage space'''
        surf2surf(self.ctx_proj_dir, self.sub_name, 'fsaverage',
                    lh_sub_file, rh_sub_file, lh_fsa_file, rh_fsa_file)
