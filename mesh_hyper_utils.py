import sys
sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/yhypergraphy")

from hyperconnectome_patch3 import load_sub_embedding_data_kernel
from hyperconnectome import load_obj_vertices
import copy
import scipy
from mesh_utils import find_k_ring_neighbors
import pickle


sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/ymeshpak")
sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/ydiffpak")

from utils import *
from dmri_data import diffusion_data

from dmri_surf_data import freesurf_data, ctx_diffusion_data
from mesh_utils import decimation_obj, read_decimatrix, read_pobj
from convert_ras_tools import convert_dwi_ras
from dmri_ctx_feature2 import gen_cnoddi0

import nibabel as nib
from mesh_freesurf_utils import load_mgh_scalers, load_freesurfer_annot_n
from collections import defaultdict
# from mesh_hyper_utils import *

def sub_clustering(X, n_clusters):
    '''Kmeans Clustering'''
    
    # clustering
    from sklearn.cluster import KMeans
    import numpy as np
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=None, init='k-means++', n_init="auto").fit(X)
    kmeans.labels_
    kclusters = kmeans.predict(X)

    # assign cluster back to surface
    kclusters = kclusters+1 # shift for unknown
    print('Expected 1:', np.min(kclusters))
    
    return kclusters

def veridx_remapping_new(kclusters, vertices_dict, sub_v_num=None):
    '''Add non-touched vertices to cluster array, the non-touched vertices will be label as 0'''
    
    embedding_v_array = np.array([v for k, v in vertices_dict.items()])
    embedding_k_array = np.array([k for k, v in vertices_dict.items()])
    
    if sub_v_num is not None:
        final_kclusters = np.zeros(sub_v_num)
    else:
        final_kclusters = np.zeros(max(embedding_k_array) + 1)
        
    final_kclusters[embedding_k_array] = kclusters
    
    return final_kclusters

def load_sub_surface(lh_obj_file, rh_obj_file, DecimationNum):

    lh_obj_file, _ = decimation_obj(lh_obj_file, DecimationNum=DecimationNum)
    rh_obj_file, _ = decimation_obj(rh_obj_file, DecimationNum=DecimationNum)

    lh_mesh, rh_mesh = load_obj_vertices(lh_obj_file, rh_obj_file)

    lr_shift = 0
    v = np.vstack([lh_mesh.vertices, rh_mesh.vertices+lr_shift])
    f = np.vstack([lh_mesh.faces, rh_mesh.faces+len(lh_mesh.vertices)])
    print(30*'*')
    print(v.shape, f.shape)

    return v, f

def neighborhood_major_vote(face, cluster_res, k=1, iter_num=1, reserve_center=True, exclude_label=None):
    '''Refine the clustering label according to mesh structure'''
    
    print(len(cluster_res))
    for _ in range(iter_num):
        for index in range(len(cluster_res)):
            neibors = find_k_ring_neighbors(face, index, k, reserve_center=reserve_center)
            if exclude_label is not None:
                local_cluster_res = cluster_res[neibors]
                local_cluster_res = local_cluster_res[local_cluster_res != exclude_label]
                if len(local_cluster_res) > 0:
                    cluster_res[index] = scipy.stats.mode(local_cluster_res)[0][0]
            else:
                cluster_res[index] = scipy.stats.mode(cluster_res[neibors])[0][0]
            
    return cluster_res

def neighborhood_major_vote_reduce_tar(face, cluster_res, tar_cluster_index=None, exclude_label=-1, k=1, iter_num=1, reserve_center=False):
    '''Refine the clustering label by replace the tar_cluster_index by neighborhood'''

    if tar_cluster_index is not None:
        for _ in range(iter_num):
            tar_ctx_indexs = np.where(cluster_res == tar_cluster_index)[0]
            print('neighborhood_major_vote_reduce_tar')
            print(len(tar_ctx_indexs))
            if len(tar_ctx_indexs) == 0:
                break
            
            for index in tar_ctx_indexs:
                neibors = find_k_ring_neighbors(face, index, k, reserve_center=reserve_center)
                local_cluster_res = cluster_res[neibors]
                local_cluster_res = local_cluster_res[local_cluster_res != tar_cluster_index]
                local_cluster_res = local_cluster_res[local_cluster_res != exclude_label]

                if len(local_cluster_res) > 0:
                    cluster_res[index] = scipy.stats.mode(local_cluster_res)[0][0]

            
    return cluster_res

def cluster_reidx(kclusters):
    '''reindex the cluster from 0 to max (some idx in kclusters are missing after refining)'''
    kclusters = kclusters.astype(int)
    kclusters_index_list = list(set(kclusters.astype(int)))
    kcluster_map = dict(zip(sorted(kclusters_index_list), list(range(len(kclusters_index_list)))))
    kclusters_remapped = [kcluster_map[idx] for idx in kclusters]
    return np.array(kclusters_remapped)
    
def load_sub_embedding_data_kernel(embedding_file, hypergraph_file):
    '''Load the subject hypergraph and embedding'''
    
    # load embedding
    sub_embedding = np.load(embedding_file)
    
    # load hyperedges
    with open(hypergraph_file, 'rb') as file:  # Python 3: open(..., 'rb')
        hyperedges = pickle.load(file)
    
    # feature index 2 real vertices index mapping (inv_node_dict)
    unique_node_idexs = np.sort(np.unique(np.concatenate([np.array(edge) for edge in hyperedges])))
    sequential_node_idexs = np.array(range(len(unique_node_idexs)))
    node_dict = dict(zip(unique_node_idexs, sequential_node_idexs))
    inv_node_dict = dict(zip(sequential_node_idexs, unique_node_idexs))    
    
    # validate data
#     print(len(sub_embedding), len(sub_embedding) == len(unique_node_idexs))
    
    return sub_embedding, node_dict

def remove_small_cluster(cluster_labels0, assign_label=-2, cluster_size_thr=10):


    cluster_labels = copy.deepcopy(cluster_labels0)
    cluster_label_set = list(np.unique(cluster_labels))
    cluster_sizes = [np.sum(cluster_labels == i) for i in cluster_label_set]
    for cluster_label, cluster_size in zip(cluster_label_set, cluster_sizes):
        if (cluster_size < cluster_size_thr) and (cluster_label != 0):
            cluster_labels[cluster_labels==cluster_label] = assign_label

    return cluster_labels

###################################### Evaluation CHI
from sklearn.metrics import calinski_harabasz_score

def cal_CHI(feature, cluster_data):
    '''Calinski-Harabasz index (CHI) '''
    cluster_labels = np.unique(cluster_data)
    
    def calculate_centroid(dataset):
        return np.mean(dataset, axis=0)
    
    #BCSS (Between-Cluster Sum of Squares)
    
    feature_clusters = [feature[cluster_data==cluster_label] for cluster_label in cluster_labels]
    feature_clusters_mean = np.array([np.mean(feature_cluster) for feature_cluster in feature_clusters])
    feature_clusters_var = np.array([np.std(feature_cluster)**2 for feature_cluster in feature_clusters])
    feature_clusters_size = np.array([len(feature_cluster) for feature_cluster in feature_clusters])

    mu = np.mean(feature)
    n = len(feature)
    k = len(cluster_labels)
#     print(mu, n, k)

    BCSS = np.sum(feature_clusters_size*(feature_clusters_mean - mu)**2)    
#     WCSS = np.sum(feature_clusters_var)
    WCSS = np.sum(feature_clusters_size*feature_clusters_var)

    cali = (n-k)/(k-1)
#     print(BCSS, WCSS, cali)
    CH = cali*BCSS/WCSS
    
    return BCSS, WCSS, CH

def exclude_unknow_annot(feat_data, annot_data, unkown_label):
    
    return feat_data[annot_data != unkown_label], annot_data[annot_data != unkown_label]  



def decimate_annot(deci_matrix, annots, dc_annot_file=None):
    '''Generate dcimate annotation'''
    
    if (dc_annot_file is not None) and os.path.exists(dc_annot_file):
        print(f'Loading dc_annot_file: {dc_annot_file}')
        with open(dc_annot_file, 'rb') as f:
             max_values = np.load(f)
#             max_values = np.load(dc_annot_file)
    else:
        max_values = []
        for dc_slice0 in deci_matrix:
            dc_slice = dc_slice0.todense()[0].tolist()[0]
        #     print(len(dc_slice))
            d = defaultdict(int)
            for weight, annot in zip(dc_slice, annots):
                d[annot] += weight
            max_value = max(d, key=lambda k: d[k])
            max_values.append(max_value)

    return max_values


def extract_local_mean_feat(feat_data, annot_data):
    
    local_feat = {}
    for label in np.unique(annot_data):
        local_feat[label] = np.mean(feat_data[annot_data == label])
        
    return local_feat


def load_surf_feature(surf_obj_file, annot_file, feature_file, DecimationNum=None):
    from mesh_utils import Mesh

    xh_vs, xh_faces = read_pobj(surf_obj_file)
    xh_annot, _, _ = nib.freesurfer.io.read_annot(annot_file)
    xh_mesh = Mesh(xh_vs, xh_faces)
    xh_mesh.annot = xh_annot

    if feature_file.split('.')[-1] == 'thickness':
        feat_data = nib.freesurfer.io.read_morph_data(feature_file)
    elif feature_file.split('.')[-1] == 'mgh':
        feat_data = load_mgh_scalers(feature_file)

    xh_mesh.feat = feat_data

    if DecimationNum is not None:
        print('Load Decimated Res!')
        # load dc mesh
        xh_obj_dc_file, xh_raw_dc_file = decimation_obj(surf_obj_file, DecimationNum=DecimationNum)
        xh_dc_vs, xh_dc_faces = read_pobj(xh_obj_dc_file)
        xh_raw_dc = read_decimatrix(xh_raw_dc_file)

        # generate dc annot 
        xh_annot_pfix = os.path.basename(annot_file)
        xh_dc_annot_file = xh_raw_dc_file.replace('.raw', f'_{xh_annot_pfix}.npy')
        xh_dc_annot = decimate_annot(xh_raw_dc, xh_annot, dc_annot_file=xh_dc_annot_file)

        xh_dc_mesh = Mesh(xh_dc_vs, xh_dc_faces)
        xh_dc_mesh.annot = xh_dc_annot

        # project feature to deci
        xh_dc_feat_data = feat_data@xh_raw_dc.T
        xh_dc_mesh.feat = xh_dc_feat_data

        return xh_dc_mesh, xh_dc_feat_data
        
    else:
        return xh_mesh, feat_data
    
    
def gen_ctx_feat(_fdata, _free_diff_data, hemi_name, mesh_name='dc_mesh', DecimationNum=0.1):

    surf_file = getattr(_fdata, f'{hemi_name}_obj_surf_file')
    annot_file = getattr(_fdata, f'{hemi_name}_annot_file')
    thick_file = getattr(_fdata, f'{hemi_name}_thick_file')

    print(annot_file)
    print(thick_file)
    print(surf_file)

    dc_mesh, thick = load_surf_feature(surf_file, annot_file, thick_file, DecimationNum=DecimationNum)
    setattr(dc_mesh, 'thick', thick)

    _free_diff_data_xh = getattr(_free_diff_data, f'{hemi_name}')
    # for feat_name in ['fa', 'md', 'odi', 'ndi']:
    for feat_name in ['fa', 'md']:
        print(getattr(_free_diff_data_xh, f'{feat_name}_file'))
        _ , feat = load_surf_feature(surf_file, annot_file, getattr(_free_diff_data_xh, f'{feat_name}_file'), DecimationNum=DecimationNum)
        setattr(dc_mesh, feat_name, feat)
        
        feat_annot_mean = extract_local_mean_feat(getattr(dc_mesh, feat_name), dc_mesh.annot)
#         feat_annot_mean = extract_local_mean_feat(feat, dc_mesh.annot)
        setattr(dc_mesh, f'{feat_name}_annot_mean', feat_annot_mean)

    setattr(_free_diff_data_xh, mesh_name, dc_mesh)

