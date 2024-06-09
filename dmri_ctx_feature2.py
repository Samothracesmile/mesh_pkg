import sys
sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/ydiffpak")
from diff_utils import *

from utils import *
from mesh_freesurf_utils import load_mgh_scalers, save_mgh_scalers

from dipy.core.gradients import gradient_table

def gen_cnoddi0(diffdata, odi_file, ndi_file, cpu_num=1, delta=0.0106, Delta=0.0431, b0_thr=80, force_flg=False):
    
    if not exists(odi_file) or force_flg:
        from dmipy.core.acquisition_scheme import gtab_dipy2dmipy
        from dmipy.signal_models import cylinder_models, gaussian_models
        from dmipy.distributions.distribute_models import SD1WatsonDistributed
        from dmipy.core.modeling_framework import MultiCompartmentModel

        # merge two shell dwi data
        dwi_file, bval_file, bvec_file = diffdata.dwi_file, diffdata.bval_file, diffdata.bvec_file

        dwi_img = load_mgh_scalers(dwi_file)
        bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)


        gtab_dipy = gradient_table(bvals, bvecs, b0_threshold=b0_thr, big_delta=Delta, small_delta=delta)
        acq_scheme_mipy = gtab_dipy2dmipy(gtab_dipy, b0_threshold=b0_thr*1e6)
        acq_scheme_mipy.print_acquisition_info

        # noddi models
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()
        zeppelin = gaussian_models.G2Zeppelin()

        watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
        watson_dispersed_bundle.parameter_names

        watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
        watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
        watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)

        NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
        NODDI_mod.parameter_names
        NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)

        NODDI_fit_hcp = NODDI_mod.fit(acq_scheme_mipy, 
                        dwi_img, mask=np.sum(dwi_img,axis=1)>0, number_of_processors=cpu_num)

        # # get odi
        odi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
        # # get ndi
        ndi = NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']

        # # get total Stick signal contribution
        # vf_intra = (NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] *
        #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])

        # # get total Zeppelin signal contribution
        # vf_extra = ((1 - NODDI_fit_hcp.fitted_parameters['SD1WatsonDistributed_1_partial_volume_0']) *
        #             NODDI_fit_hcp.fitted_parameters['partial_volume_1'])

        save_mgh_scalers(odi, diffdata.fa_file, odi_file)
        save_mgh_scalers(ndi, diffdata.fa_file, ndi_file)    
