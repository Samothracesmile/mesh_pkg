'''
Calculate dMRI features:  DTI, RISH, Moment

'''
import sys
sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/ydiffpak")
from diff_utils import *

from utils import *
from mesh_freesurf_utils import load_mgh_scalers, save_mgh_scalers
from dipy.data import default_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import peaks_from_model

def gen_cdti(dwi_file, bval_file, bvec_file, 
            fa_file, md_file, gfa_file, 
            b0_thr=80, tar_bval=None, force_flg=False):

    if not (exists(fa_file) and exists(md_file) and exists(gfa_file)) or force_flg:
        # load dwi data
        dwi = load_mgh_scalers(dwi_file)
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)

        # extract tar bval data
        if tar_bval is not None:
            b0_mask = bvals < b0_thr
            tar_bval_mask = (bvals < (tar_bval+100)) & (bvals > (tar_bval-100))

            dwi = np.concatenate([dwi[..., b0_mask], dwi[..., tar_bval_mask]], axis=1)
            bvals = np.concatenate([bvals[b0_mask], bvals[tar_bval_mask]])
            bvecs = np.concatenate([bvecs[b0_mask], bvecs[tar_bval_mask]])

        print(bvals)

        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)

        # fa and md
        dtimodel = reconst_dti.TensorModel(gtab, fit_method ="LS")
        dtifit = dtimodel.fit(dwi)
        save_mgh_scalers(dtifit.fa, dwi_file, fa_file)
        save_mgh_scalers(dtifit.md, dwi_file, md_file)

        # gfa
        csamodel = CsaOdfModel(gtab, 6) # using 6 order 
        csapeaks = peaks_from_model(model=csamodel,
                            data=dwi,
                            sphere=default_sphere,
                            relative_peak_threshold=.5,
                            min_separation_angle=25,
                            return_odf=False,
                            normalize_peaks=True)

        gfa_data = csapeaks.gfa
        save_mgh_scalers(gfa_data, dwi_file, gfa_file)

def gen_crish(dwi_file, bval_file, bvec_file,
            rish_file_pattern, sh_num, 
            b0_thr=80, force_flg=False):
    
    '''
    Generate RISH images
    
    '''
    
    # 1. Check if expected rish files exists  
    rish_exist_flg = len(glob(rish_file_pattern)) >= (sh_num//2 + 1)
    
    # 2. calculate RISH features
    if not rish_exist_flg or force_flg:
        dwi = load_mgh_scalers(dwi_file)
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        
        # form a full sphere
        bvals = np.append(bvals, bvals)
        bvecs = np.append(bvecs, -bvecs, axis=0)
        dwi = np.append(dwi, dwi, axis=-1)
        
        dwi_norm, _ = normalize_data(dwi, where_b0=np.where(bvals < b0_thr)[0])
        
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_thr)
        qb_model = QballModel(gtab, sh_order=sh_num)
        
        # inserting correct shm_coeff computation block ---------------------------------
        smooth = 0.00001

        L = qb_model.n*(qb_model.n+1)
        L**=2
        _fit_matrix = np.linalg.pinv(qb_model.B.T @ qb_model.B + np.diag(smooth*L)) @ qb_model.B.T
        shm_coeff = np.dot(dwi_norm[..., qb_model._where_dwi], _fit_matrix.T)
        # shm_coeff = applymask(shm_coeff, mask)
        # -------------------------------------------------------------------------------

        shm_coeff_squared = shm_coeff**2
        shs_same_level = [[0, 1], [1, 6], [6, 15], [15, 28], [28, 45]]
        
        shs_files = []
        for i in range(0, sh_num+1, 2):
            ind = int(i/2)
            temp = np.sum(shm_coeff_squared[...,shs_same_level[ind][0]:shs_same_level[ind][1]], axis=-1)
            shs_file = rish_file_pattern.replace('*',f'{ind*2}')
            shs_files.append(shs_file)
            save_mgh_scalers(temp, dwi_file, shs_file)
    else:
        shs_files = []
        for i in range(0, sh_num+1, 2):
            ind = int(i/2)
            shs_file = rish_file_pattern.replace('*',f'{ind*2}')
            assert exists(shs_file)
            shs_files.append(shs_file)
        
    return shs_files

def gen_cmoment(dwi_file, bval_file, bvec_file,
            moment_file_pattern, 
            b0_thr=80, force_flg=False):
    
    '''
    Generate dMRI Moment images (first order moment and Second Centrial Moment)
    
    '''
    
    # 1. Check if expected rish files exists  
    moment_files = sorted(glob(moment_file_pattern))
    moment_exist_flg = len(moment_files) == 2
    
    # 2. calculate Moments
    if not moment_exist_flg or force_flg:
        dwi = load_mgh_scalers(dwi_file)
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
        
        # form a full sphere
        bvals = np.append(bvals, bvals)
        bvecs = np.append(bvecs, -bvecs, axis=0)
        dwi = np.append(dwi, dwi, axis=-1)
        
        # normalized dMRI
        dwi_norm, _ = normalize_data(dwi, where_b0=np.where(bvals < b0_thr)[0])
        dwi_norm_b1 = dwi_norm[...,bvals > b0_thr] # extract the non b0 diffusion signal

        # estimate two moments
        org_moment1 = np.mean(dwi_norm_b1,axis=-1)
        org_moment2 = np.sum((dwi_norm_b1 - org_moment1[...,None])**2, axis=-1)/(dwi_norm_b1.shape[-1] - 1) 
        
        moment1_file = moment_file_pattern.replace('*', str(1))
        moment2_file = moment_file_pattern.replace('*', str(2))
        
        save_mgh_scalers(org_moment1, dwi_file, moment1_file)
        save_mgh_scalers(org_moment2, dwi_file, moment2_file)
        
        moment_files = [moment1_file, moment2_file]
    
    return moment_files



# from dipy.core.gradients import gradient_table
# from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
# from dmipy.signal_models import cylinder_models, gaussian_models
# from dmipy.distributions.distribute_models import SD1WatsonDistributed
# from dmipy.core.modeling_framework import MultiCompartmentModel


# def gen_cnoddi0(diffdata, odi_file, ndi_file, cpu_num=1, delta=0.0106, Delta=0.0431, b0_thr=80, force_flg=False):
    
#     if not exists(odi_file) or force_flg:
#         # merge two shell dwi data
#         dwi_file, bval_file, bvec_file = diffdata.dwi_file, diffdata.bval_file, diffdata.bvec_file

#         dwi_img = load_mgh_scalers(dwi_file)
#         bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)


#         gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_thr, big_delta=Delta, small_delta=delta)
#         acq_scheme_mipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_thr*1e6)
#         acq_scheme_mipy.print_acquisition_info

#         # noddi models
#         ball = gaussian_models.G1Ball()
#         stick = cylinder_models.C1Stick()
#         zeppelin = gaussian_models.G2Zeppelin()

#         watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
#         watson_dispersed_bundle.parameter_names

#         watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
#         watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
#         watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

#         NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
#         NODDI_mod.parameter_names
#         NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

#         NODDI_fit_hcp = NODDI_mod.fit(acq_scheme_mipy, 
#                         dwi_img, mask=np.sum(dwi_img,axis=1)>0, number_of_processors=cpu_num)

#         # # get odi
#         odi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
#         # # get ndi
#         ndi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']

#         # # get total Stick signal contribution
#         # vf_intra = (NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] *
#         #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])

#         # # get total Zeppelin signal contribution
#         # vf_extra = ((1 - NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) *
#         #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])

#         save_mgh_scalers(odi, diffdata1.fa_file, odi_file)
#         save_mgh_scalers(ndi, diffdata1.fa_file, ndi_file)    




# def gen_cnoddi(diffdata1, diffdata2, odi_file, ndi_file, cpu_num=1, delta=0.0106, Delta=0.0431, b0_thr=80, force_flg=False):
    
#     if not exists(odi_file) or force_flg:
#         # merge two shell dwi data
#         dwi_file1, bval_file1, bvec_file1 = diffdata1.dwi_file, diffdata1.bval_file, diffdata1.bvec_file
#         dwi_file2, bval_file2, bvec_file2 = diffdata2.dwi_file, diffdata2.bval_file, diffdata2.bvec_file

#         dwi_img1 = load_mgh_scalers(dwi_file1)
#         dwi_img2 = load_mgh_scalers(dwi_file2)    

#         bvals1, bvecs1 = read_bvals_bvecs(bval_file1, bvec_file1)
#         bvals2, bvecs2 = read_bvals_bvecs(bval_file2, bvec_file2)

#         dwi_img = np.concatenate([dwi_img1.T, dwi_img2.T]).T
#         bvals = np.concatenate([bvals1, bvals2])
#         bvecs = np.concatenate([bvecs1, bvecs2])

#         gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_thr, big_delta=Delta, small_delta=delta)
#         acq_scheme_mipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_thr*1e6)
#         acq_scheme_mipy.print_acquisition_info

#         # noddi models
#         ball = gaussian_models.G1Ball()
#         stick = cylinder_models.C1Stick()
#         zeppelin = gaussian_models.G2Zeppelin()

#         watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
#         watson_dispersed_bundle.parameter_names

#         watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
#         watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
#         watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

#         NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
#         NODDI_mod.parameter_names
#         NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

#         NODDI_fit_hcp = NODDI_mod.fit(acq_scheme_mipy, 
#                         dwi_img, mask=np.sum(dwi_img,axis=1)>0, number_of_processors=cpu_num)

#         # # get odi
#         odi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
#         # # get ndi
#         ndi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']

#         # # get total Stick signal contribution
#         # vf_intra = (NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] *
#         #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])

#         # # get total Zeppelin signal contribution
#         # vf_extra = ((1 - NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) *
#         #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])

#         save_mgh_scalers(odi, diffdata1.fa_file, odi_file)
#         save_mgh_scalers(ndi, diffdata1.fa_file, ndi_file)    


# def gen_cnoddi2(diffdata1, diffdata2, odi_file, ndi_file, cpu_num=1, delta=0.0106, Delta=0.0431, b0_thr=80, force_flg=False):
    
#     if not exists(odi_file) or force_flg:
#         # merge two shell dwi data
#         dwi_file1, bval_file1, bvec_file1 = diffdata1.dwi_file, diffdata1.bval_file, diffdata1.bvec_file
#         dwi_file2, bval_file2, bvec_file2 = diffdata2.dwi_file, diffdata2.bval_file, diffdata2.bvec_file

#         dwi_img1 = load_mgh_scalers(dwi_file1)
#         dwi_img2 = load_mgh_scalers(dwi_file2)    

#         bvals1, bvecs1 = read_bvals_bvecs(bval_file1, bvec_file1)
#         bvals2, bvecs2 = read_bvals_bvecs(bval_file2, bvec_file2)

#         dwi_img = np.concatenate([dwi_img1.T, dwi_img2.T]).T
#         bvals = np.concatenate([bvals1, bvals2])
#         bvecs = np.concatenate([bvecs1, bvecs2])

#         gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_thr, big_delta=Delta, small_delta=delta)
#         acq_scheme_mipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_thr*1e6)
#         acq_scheme_mipy.print_acquisition_info

#         # noddi models
#         ball = gaussian_models.G1Ball()
#         stick = cylinder_models.C1Stick()
#         zeppelin = gaussian_models.G2Zeppelin()

#         watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
#         watson_dispersed_bundle.parameter_names

#         watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
#         watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
#         watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.1e-9)

#         NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
#         NODDI_mod.parameter_names
#         NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

#         NODDI_fit_hcp = NODDI_mod.fit(acq_scheme_mipy, 
#                         dwi_img, mask=np.sum(dwi_img,axis=1)>0, number_of_processors=cpu_num)

#         # # get odi
#         odi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
#         # # get ndi
#         ndi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']

#         # # get total Stick signal contribution
#         # vf_intra = (NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] *
#         #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])

#         # # get total Zeppelin signal contribution
#         # vf_extra = ((1 - NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) *
#         #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])

#         save_mgh_scalers(odi, diffdata1.fa_file, odi_file)
#         save_mgh_scalers(ndi, diffdata1.fa_file, ndi_file)    