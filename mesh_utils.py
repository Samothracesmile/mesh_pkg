import os
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix



def decimation_obj(obj_file, out_obj_file=None, out_raw_file=None, DecimationNum=0.25, force_flg=False):
    '''Downsample surface'''
    
    if out_obj_file is None:
        out_obj_file = obj_file.replace('.obj', f'_dm{int(100*DecimationNum)}.obj')
        print(f'out_obj_file: {out_obj_file}.')
    if out_raw_file is None:
        out_raw_file = obj_file.replace('.obj', f'_dm{int(100*DecimationNum)}_DecimationMtx.raw')    
        print(f'out_raw_file: {out_raw_file}.')
        
    # run mesh decimation
    mesh_decimation_str = '/ifs/loni/faculty/shi/spectrum/yshi/MeshEigenAnalysis/Executables/MeshDecimation'
    mesh_decimation_cmd = f'{mesh_decimation_str} {obj_file} 1 {DecimationNum} 0 10 {out_obj_file} {out_raw_file}'
    if (not os.path.exists(out_obj_file)) or force_flg:
        os.system(mesh_decimation_cmd)
    
    return out_obj_file, out_raw_file



def find_k_ring_neighbors(faces, index, k, reserve_center=False):
    
    def find_1_ring_neighbors(faces, index, reserve_center=False):
        '''detect 1 ring neighbors'''
    
        ring_nei = np.unique(faces[np.any(faces == index, axis=1)]).tolist()
        if not reserve_center:
            ring_nei.remove(index)

        return np.array(ring_nei)

    tar_idexs = np.array([index])
    
    for _ in range(k):
        tar_idexs = np.concatenate([find_1_ring_neighbors(faces, index, reserve_center=False) for index in tar_idexs])
        tar_idexs = np.unique(tar_idexs)
        selfmask = tar_idexs == index
        tar_idexs = tar_idexs[~selfmask]
        
    if reserve_center:
        tar_idexs = np.concatenate([np.array([index]), tar_idexs])

    return tar_idexs


def read_mni_obj(filename):

    with open(filename, 'r') as f:
        data = f.readlines()

        s = '\n'.join(data)
        data = s.split()

        if data[0] != 'P':
            raise ValueError('Only Polygons supported')

        surfprop = [float(data[1]), float(data[2]), float(data[3]), int(data[4]), int(data[5])]
        n_points = int(data[6])

        start = 7
        end = n_points * 3 + start
        point_array = [np.float32(x) for x in data[start:end]]
        point_array = np.reshape(point_array, (n_points, 3,))

        start = end
        end = n_points * 3 + start
        normals = [np.float32(x) for x in data[start:end]]
        normals = np.reshape(normals, (n_points, 3,))

        nitems = int(data[end])
        colour_flag = int(data[end + 1])

        start = end + 2
        end = start + 4
        colour_table = np.array(data[start:end]).astype('float')

        start = end + 4
        end = start + 2*nitems

        start = end
        end = start + nitems
        end_indices = [int(i) for i in data[start:end]]

        start = end
        end = start + end_indices[-1] + 1
        indices = [int(i) for i in data[start:end]]
        faces = np.reshape(indices, (-1, 3,))

        header_info = surfprop, n_points, nitems, colour_flag, colour_table, end_indices
        
    return header_info, point_array, normals, faces

def list2str(array, two_deci=True, splitter=' '):
    """
    Join a list with spaces between elements.
    """
#     return ' '.join(str(a) for a in array)
    if two_deci:
        str_list = [f'{a:.2f}' for a in array]
    else: 
        str_list = [str(a) for a in array]
        
    return splitter.join(str_list)

def save_mni_obj(filename, header_info, point_array, normals, faces):
    """
    Write this object to a file.
    """
    
    assert not os.path.exists(filename), f'Do not overwrite {filename}!'
    
    surfprop, n_points, nitems, colour_flag, colour_table, end_indices = header_info
    
    with open(filename, 'w') as file:
#         float(data[1]), float(data[2]), float(data[3]), int(data[4]), int(data[5])
        header = ['P'] + surfprop + [n_points]
        file.write(list2str(header, two_deci=False) + '\n')

        for point in point_array:
            file.write('  ' + list2str(point, splitter='  ') + '\n')

        for vector in normals:
            file.write(' ' + list2str(vector) + '\n')

        file.write(f'\n{nitems}\n')

        colour_str = ' '.join(list(colour_table.astype('str'))) + '\n'
        colour_str = colour_str*(n_points)
        file.write(f'{colour_flag} {colour_str}\n')
        
        
        for i in range(0, nitems, 8):
            file.write(' ' + list2str(end_indices[i:i + 8], two_deci=False) + '\n')

#         for i in range(0, len(faces), 8):
#             file.write(' ' + list2str(faces.flatten()) + '\n')
        file.write(f'\n')
        file.write(list2str(faces.flatten(), two_deci=False) + '\n')

def read_raw(file, dtype='double'):
    # import rawpy
#     return rawpy.imread(file)
    return np.fromfile(file, dtype=dtype)


def read_decimatrix(file):
    '''
    Read the mesh downsample transfer matrix
    '''
    Q = np.fromfile(file, dtype='double')
    Q = np.reshape(Q, (-1,3)) 
    
    return csr_matrix((Q[:,2], (Q[:,0].astype(int), Q[:,1].astype(int))), 
                        shape=(int(max(Q[:,0]+1)), int(max(Q[:,1]+1))))


def read_obj(file):
    '''Load the wavefront .obj file return vertices and faces as numpy arraries'''

    vs, faces = [], []

    with open(file) as f:    
        for line in f:
        #     print()
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                                   for ind in face_vertex_ids]
                faces.append(face_vertex_ids)

    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()

    return vs, faces



def write_obj(filepath, vertices, triangles, normals=None):
    """Writes a Mesh3D object out to a .obj file format
    Parameters
    ----------
    mesh : :obj:`Mesh3D`
        The Mesh3D object to write to the .obj file.
    Note
    ----
    Does not support material files or texture coordinates.
    """
    f = open(filepath, 'w')

    # write human-readable header
    f.write('###########################################################\n')
    f.write('# OBJ file generated by UC Berkeley Automation Sciences Lab\n')
    f.write('#\n')
    f.write('# Num Vertices: %d\n' %(vertices.shape[0]))
    f.write('# Num Triangles: %d\n' %(faces.shape[0]))
    f.write('#\n')
    f.write('###########################################################\n')
    f.write('\n')

    for v in vertices:
        f.write('v %f %f %f\n' %(v[0], v[1], v[2]))

    # write the normals list
    if normals is not None and normals.shape[0] > 0:
        for n in normals:
            f.write('vn %f %f %f\n' %(n[0], n[1], n[2]))

    # write the normals list
    for t in faces:
        f.write('f %d %d %d\n' %(t[0]+1, t[1]+1, t[2]+1)) # convert back to 1-indexing

    f.close()

# Mesh = namedtuple('Mesh', 'vertices faces scalars')

from collections import namedtuple
# Create a namedtuple object for meshes
SimpleMesh = namedtuple("mesh", ["coordinates", "faces"])
# Create a namedtuple object for surfaces
SimpleSurface = namedtuple("surface", ["mesh", "data"])

class Mesh():
    '''Mesh'''
    
    def __init__(self, vertices, faces, scalars=None):
        self.faces = faces
        self.vertices = vertices
        self.scalars = scalars    

    def plot(self,scalars_name='None'):
        if self.scalars is None:
            self.scalars = np.ones(self.vertices.shape[0])
            
        plot_mesh(self.vertices,self.faces,self.scalars,scalars_name)
        
        
    def get_scalars(self, scalars):
        self.scalars = scalars



class FsaverageHemi():
    '''Fsaverage Surface, Label, and Features'''
    
    def __init__(self, faces, vertices):
        self.faces = faces
        self.vertices = vertices
        
    def get_annot(self, annot_struct_names,
                  annot_struct_labels,annot_labels):
        self.annot_struct_names = annot_struct_names
        self.annot_struct_labels = annot_struct_labels
        self.annot_labels = annot_labels

        self.roi_dict = dict(zip(self.annot_struct_names,range(len(self.annot_struct_names))))


    def show_annot_names(self,annot_idx_list=[]):
        if annot_idx_list:
            region_names = [self.annot_struct_names[annot_idx] for annot_idx in annot_idx_list]
            print(f'{region_names}')
        else:
            for i,annot_name in enumerate(self.annot_struct_names):
                print(f'{i}:{annot_name}')

    def get_decimtx(self,decimtx):
        self.decimtx = decimtx

    def extract_region_mask(self, roi_name):
        annot_idx = self.roi_dict[roi_name]
        annot_mask = self.annot_labels == self.annot_struct_labels[annot_idx]

        return annot_mask

    def extract_rois_suvrs(self,ctx_roi_list,suvrs):
        rois_suvrs_list = []
        for roi_name in ctx_roi_list:
            roi_mask = self.extract_region_mask(roi_name)

            roi_suvrs = []
            for suvr in suvrs:
                roi_suvrs.append(np.mean(suvr[roi_mask]))

            rois_suvrs_list.append(roi_suvrs)
        return np.array(rois_suvrs_list)

    def extract_rois_suvr(self,ctx_roi_list,suvr):
        rois_suvr_list = []
        for roi_name in ctx_roi_list:
            roi_mask = self.extract_region_mask(roi_name)

            rois_suvr_list.append(np.mean(suvr[roi_mask]))
        return np.array(rois_suvr_list)

    def extract_region(self,annot_idx,r_annot_thr=0.01,face_touch_mode='all',verbose=False):
        if not isinstance(annot_idx, list):
            region_name = self.annot_struct_names[annot_idx]
            
            if verbose:
                print(f'Extracting region {region_name} with annot_thr = {r_annot_thr}')
                
            annot_mask = self.annot_labels == self.annot_struct_labels[annot_idx] # extract the vertices annot mask
        else:
            annot_mask_list = []
            for a_annot_idx in annot_idx:
                region_name = self.annot_struct_names[annot_idx]
                
                if verbose:
                    print(f'Extracting region {region_name} with annot_thr = {r_annot_thr}')
                    
                a_annot_mask = self.annot_labels == self.annot_struct_labels[annot_idx] # extract the vertices annot mask
                annot_mask_list.append(np.expand_dims(a_annot_mask, axis=0))
            annot_mask_array = np.concatenate(annot_mask_list, axis=0)
            annot_mask = np.any(annot_mask_array, axis=0)
            
        r_annot_mask = self.decimtx*annot_mask > r_annot_thr

        r_annot_mask_index_set = set(r_annot_mask.nonzero()[0])
        r_faces_mask = np.array([v in r_annot_mask_index_set for v in self.faces.flatten()]).reshape(self.faces.shape)
        
        # involve the faces with any/all corresponding vertices
        if face_touch_mode == 'any':
            r_faces = self.faces[np.any(r_faces_mask,axis=1)]
        else:
            r_faces = self.faces[np.all(r_faces_mask,axis=1)]

        r_annot_face_vertices_list = sorted(list(set(r_faces.flatten())))

        # reindex the faces
        mapdict = dict(zip(r_annot_face_vertices_list, range(len(r_annot_face_vertices_list))))
        r_faces = np.array([mapdict[x] for x in r_faces.flatten()]).reshape(r_faces.shape)

        # extract the vertices
        r_vertices = self.vertices[r_annot_face_vertices_list]
        
        return r_vertices,r_faces,r_annot_face_vertices_list
    
    def extract_region_suvr(self,annot_idx,sub_suvr,r_annot_thr=0.1):
        r_vertices,r_faces,r_annot_face_vertices_list = self.extract_region(annot_idx,r_annot_thr)
        r_sub_suvr = sub_suvr[r_annot_face_vertices_list]
        
        return r_vertices,r_faces,r_sub_suvr

    def extract_region_mesh(self,annot_idx,sub_suvr,r_annot_thr=0.1):
        r_vertices,r_faces,r_annot_face_vertices_list = self.extract_region(annot_idx,r_annot_thr)
        r_sub_suvr = sub_suvr[r_annot_face_vertices_list]
        
        region_mesh = Mesh(r_vertices,r_faces,r_sub_suvr)
        
        return region_mesh    

    
def load_fsaverage_from_mat(mat_fname):
    '''Load the hardcoded fsaverage mesh, annotation, and downsampling matrix'''
    import scipy.io as sio
    mat_contents = sio.loadmat(mat_fname)
#     sorted(mat_contents.keys()) # print the mat contents
    decimtx = mat_contents['decimtx']
    faces = mat_contents['faces'] - 1 # convert the matlab index to python index     
    vertices = mat_contents['vertices']    
    
    annot_struct_names = [names[0][0] for names in mat_contents['annot_struct_names']]
    annot_struct_labels = mat_contents['annot_struct_labels'].flatten()
    annot_labels = mat_contents['annot_label'].flatten()
    
    hemi = FsaverageHemi(faces, vertices)
    hemi.get_annot(annot_struct_names,annot_struct_labels,annot_labels)
    hemi.get_decimtx(decimtx)
    
    return hemi


def laplace_filter(eig_funcs,suvr):
    '''
    eig_funcs(np.array)
    suvr(np.array)
    
    output: filtered_suvr(np.array)
    '''
    return np.matmul(eig_funcs,np.matmul(eig_funcs.T, suvr))
    
def laplace_filter_suvrs(eig_fname, suvrs):
    '''
    eig_fname(str): The .mat file contrains V as the eig_vectors
    suvrs(list): The list of corresponding suvr array
    
    output: filtered suvrs(list)
    '''
    print('Applying laplace filtering')
    import scipy.io as sio
    mat_contents = sio.loadmat(eig_fname)
    eig_funcs = mat_contents['V']

    return [laplace_filter(eig_funcs,suvr) for suvr in suvrs]
    



def plot_mesh(points,triangles,scalars,scalars_name='None'):
    
    from tvtk.api import tvtk
    from mayavi.scripts import mayavi2

    # The TVTK dataset.
    mesh = tvtk.PolyData(points=points, polys=triangles)
    mesh.point_data.scalars = scalars
    mesh.point_data.scalars.name = scalars_name

    # Uncomment the next two lines to save the dataset to a VTK XML file.
    #w = tvtk.XMLPolyDataWriter(input=mesh, file_name='polydata.vtp')
    #w.write()

    # Now view the data.
    @mayavi2.standalone
    def view():
        from mayavi.sources.vtk_data_source import VTKDataSource
        from mayavi.modules.surface import Surface

        mayavi.new_scene()
        src = VTKDataSource(data = mesh)
        mayavi.add_source(src)
        s = Surface()
        mayavi.add_module(s)

    view()       


def extract_adni_subname(fname):
    base = os.path.basename(fname)
    return base[:10]

def extract_adni_subname_hemi(fname):
    base = os.path.basename(fname)
    return base[:13]

def merge_meshes(mesh_list):
    # reindex the faces
    vertices_num_list = [0]+[len(mesh.vertices) for mesh in mesh_list[:-1]]
    vertices_cumsums = list(np.cumsum(vertices_num_list))
    reindexed_face_list = [mesh.faces + cumsum for mesh,cumsum in zip(mesh_list,vertices_cumsums)]
    
    # concatenate
    reindexed_faces = np.concatenate(reindexed_face_list,axis=0)
    vertices = np.concatenate([mesh.vertices for mesh in mesh_list],axis=0)
    scalars = np.concatenate([mesh.scalars for mesh in mesh_list],axis=0)
    
    return Mesh(vertices, reindexed_faces, scalars)



# mesh to graph

def gen_adjmatrix(mesh):
    vert_num = len(mesh.vertices)
    adj_mat = np.zeros([vert_num,vert_num]) > 100

    for face in mesh.faces:
        aaa,bbb,ccc = face
        adj_mat[aaa,bbb] = True
        adj_mat[ccc,bbb] = True
        adj_mat[aaa,ccc] = True
    
    # ensure symmetric
    adj_mat_sum = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    
    return adj_mat

def convert_mesh2graph(mesh):
    
    # generate the networkx graph from the mesh adj_mat
    adj_mat = gen_adjmatrix(mesh)
    G = nx.from_numpy_matrix(adj_mat)
    # asign the suvr as the node feature
    nx.set_node_attributes(G, dict(enumerate(mesh.scalars)),'feat')
#     nx.set_node_attributes(G, dict(enumerate(np.expand_dims(merged_mesh.scalars, axis=1))),'feat')
    nx.set_node_attributes(G, dict(enumerate(mesh.vertices)),'vert')
    
    return G


def get_attri(df, subjid, *attris):
    # retrieve  the attributes from dataframe
    attris = list(attris)
    return df.loc[subjid][attris]



# read the pobj and interpolation (01/17/2021)

# def read_pobj(file):
#     '''read the Polyon .obj file return vertices and faces as numpy arraries'''

#     with open(file, 'r') as f:
#         lines = f.readlines()

#         v_num = int(lines[0].strip().split()[-1])
#         vs = np.asarray([[float(v) for v in ll.split()] for ll in lines[1:v_num+1]])
#         ns = np.asarray([[float(v) for v in ll.split()] for ll in lines[v_num+1:2*v_num+1]])

#         t_num = int(lines[2*v_num+2].strip())
#         try:
#             faces = np.asarray([int(nn) for nn in lines[-1].strip().split()]).reshape(t_num,3)
#         except:
#             faces = None

#     # assert np.logical_and(faces >= 0, faces < len(vs)).all()

#     return vs, faces



def read_pobj(file):
    '''read the Polyon .obj file return vertices and faces as numpy arraries'''

    with open(file, 'r') as f:
        lines = f.readlines()

        v_num = int(lines[0].strip().split()[-1])
        vs = np.asarray([[float(v) for v in ll.split()] for ll in lines[1:v_num+1]])
        ns = np.asarray([[float(v) for v in ll.split()] for ll in lines[v_num+1:2*v_num+1]])

        t_num = int(lines[2*v_num+2].strip())
        try:
            faces = np.asarray([int(nn) for nn in lines[-1].strip().split()]).reshape(t_num,3)
        except:
    #         faces = None
            faces_list = lines[3*v_num + 4 + t_num//8 + 1:]
            faces_str = ''.join([*faces_list])
            faces_str = faces_str.replace('\n', '')
            faces = np.asarray([int(nn) for nn in faces_str.split()]).reshape(t_num,3)

    # assert np.logical_and(faces >= 0, faces < len(vs)).all()

    return vs, faces



def read_pobj2mesh(file):
    '''read the Polyon .obj file return a mesh object'''
    vs, faces = read_pobj(file)
    
    return Mesh(vs,faces)


def sci_rgi_interp_3d(img, pts, inter_method='linear', offset=0, to_ras=False):
    from scipy.interpolate import RegularGridInterpolator as rgi
    '''
    3D interpolation using the scipy rgi interpolate
        Input: 
            3D volume (np.array)
            pts (np.array)
            
            inter_method = “linear” or “nearest”
        Output:
            scalers (np.array)
            
        !! Note: the index of pts should be comfirmed (0 or 1)
    '''
    if to_ras:
        img = img[::-1,:,:] # convert the 3d image from LAS to RAS

    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    z = np.arange(img.shape[2])
    sci_rig_interp_function = rgi((x,y,z), img, method=inter_method)

    return sci_rig_interp_function(pts-offset)
#     return sci_rig_interp_function(pts - 1)



def sci_rgi_interp_4d(dwi_img, mid_vertices_imgspc):


    mid_dwi_val = []
    for idx in range(dwi_img.shape[-1]):
        mid_dwi_val.append(sci_rgi_interp_3d(dwi_img[...,idx], 
                                mid_vertices_imgspc, inter_method='linear', offset=0.5, to_ras=True))
    mid_dwi_val = np.vstack(mid_dwi_val).T
    
    return mid_dwi_val


def vol2surf_mesh(vol_file, surf_file):
    from nibabel import load
    '''
    resample the volume data onto surface return a mesh object
    Input: 
        nifty vol_file (string)
        obj surf_file (string)
        
    Output:
        Mesh (mesh object)
        
    '''
    # 1. load surface (.obj)
    vs, faces = read_pobj(surf_file)
    
    # 2. load volume (.nii.gz)
    vol_nii = load(vol_file)
    img = vol_nii.get_fdata()
    resolution = vol_nii.affine[0,0]
    
    # 3. interpolation
    pts = vs/resolution
    scalers = sci_rgi_interp_3d(img, pts)
    
    return Mesh(vs,faces,scalers)

def vol2surf_scalers(vol_file, surf_file):
    from nibabel import load
    '''
    resample the volume data onto surface return a mesh object
    Input: 
        nifty vol_file (string)
        obj surf_file (string)
        
    Output:
        Mesh (mesh object)
        
    '''
    # 1. load surface (.obj)
    vs, faces = read_pobj(surf_file)
    
    # 2. load volume (.nii.gz)
    vol_nii = load(vol_file)
    img = vol_nii.get_fdata()
    resolution = vol_nii.affine[0,0]
    
    # 3. interpolation
    pts = vs/resolution
    scalers = sci_rgi_interp_3d(img, pts)
    
    return scalers

def voldata2surf_scalers(img, resolution, surf_file):
    from nibabel import load
    '''
    resample the volume data onto surface return a mesh object
    Input: 
        nifty vol_file (string)
        obj surf_file (string)
        
    Output:
        Mesh (mesh object)
        
    '''
    # 1. load surface (.obj)
    vs, faces = read_pobj(surf_file)
    
    # 3. interpolation
    pts = vs/resolution
    scalers = sci_rgi_interp_3d(img, pts)
    
    return scalers



def dwivol2surf_scalers(dwi_file, surf_file):
    from nibabel import load
    '''
    resample the 4d volume onto surface return scalers (numpy)
    
    Input:
        nifty 4d vol_file (string)
        obj surf_file (string)

    Output:
        Scalers (numpy array)

    '''

    # 1. load the surface (.obj)
    vs, faces = read_pobj(surf_file)

    # 2. load volume (.nii.gz)
    dwi_nii = load(dwi_file)
    dwi_imgs = dwi_nii.get_fdata()
    resolution = dwi_nii.affine[0,0]


    # 3. interpolation
    pts = vs/resolution
    scalers = []
    for i in range(dwi_imgs.shape[3]):
    #     print(dwi_imgs[...,i].shape)
        scalers.append(sci_rgi_interp_3d(dwi_imgs[...,i], pts))

    # for scaler in scalers:
    #     print(scaler.shape)

    scalers = np.array(scalers).T
    
    return scalers


# m1 save scaler as .npy
# np.save(file, arr)
# m2 load scaler from .npy