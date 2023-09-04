from anndata._core.anndata import AnnData
from torch.utils.data import Dataset
import torch
from copy import deepcopy
from scipy import spatial
from scipy.spatial.distance import cdist
import scanpy as sc
import numpy as np
from sklearn.preprocessing import StandardScaler



def sc_tsne(adata, n_pcs = 50, group = None):
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    sc.tl.tsne(adata, n_pcs)
    return adata

def distance(p1, p2):
    """ Calculate the distance between sample p1 and sample p2 """
    return np.sqrt(np.sum((p1 - p2) ** 2))

def same_side(dataset, current_point, neighbors):
    """
    Determine whether neighbors around the current sample are located on the same side of it.
    Args:
        dataset: coordinates of all samples
        current_point: coordinate of the current point
        neighbors: index list of neighbors
    Returns True if all neighbors are on the same side, otherwise returns False
    """

    vectors = dataset[neighbors] - current_point

    # compute the average vector and normalize it to unit lengths
    mean_vector = np.mean(vectors, axis=0)
    mean_vector /= np.linalg.norm(mean_vector)

    # calculate the sign ( + or - ) of the angle between all vectors and the mean vector
    angle_signs = np.sign(np.dot(vectors, mean_vector))

    # If all signs are same, it means that neighbors are on the same side, otherwise they are not on the same side.
    if np.all(angle_signs > 0) or np.all(angle_signs < 0):
        return True
    else:
        return False

def get_weights_by_density(data):
    # setting the search radius
    search_radius = 0.1

    samples = data
    scaler = StandardScaler()
    samples_norm = scaler.fit_transform(samples)

    # generate KD tree for nearest neighbor search
    tree = spatial.KDTree(samples_norm)
    neigh_num = []

    for i, sample in enumerate(samples_norm):
        neighbors = tree.query_ball_point(sample, r=search_radius)
        neighbors.remove(i)
        neigh_num.append(len(neighbors))

    weights = 1 / np.exp(neigh_num)

    return weights

def get_weights_by_distance(data):
    # setting the search radius
    search_radius = 0.1

    samples = data
    scaler = StandardScaler()
    samples_norm = scaler.fit_transform(samples)

    # generate KD tree for nearest neighbor search
    tree = spatial.KDTree(samples_norm)
    edge_points = []
    neigh_num = []

    for i, sample in enumerate(samples_norm):
        neighbors = tree.query_ball_point(sample, r=search_radius)
        neighbors.remove(i)
        neigh_num.append(len(neighbors))
        if same_side(samples_norm, sample, neighbors):
            edge_points.append(sample)

    # Calculate the distance matrix (the closest distance from each point to the edge point)
    dist_matrix = cdist(samples_norm, edge_points)
    nearest_dist = np.min(dist_matrix, axis=1)

    weights = 1 / np.exp(nearest_dist)

    return weights

def get_weights(data):
    # setting the search radius
    search_radius = 0.1

    samples = data
    scaler = StandardScaler()
    samples_norm = scaler.fit_transform(samples)

    # generate KD tree for nearest neighbor search
    tree = spatial.KDTree(samples_norm)
    edge_points = []
    neigh_num = []

    for i, sample in enumerate(samples_norm):
        neighbors = tree.query_ball_point(sample, r=search_radius)
        neighbors.remove(i)
        neigh_num.append(len(neighbors))
        if same_side(samples_norm, sample, neighbors):
            edge_points.append(sample)

    # Calculate the distance matrix
    dist_matrix = cdist(samples_norm, edge_points)
    # Get the closest distance for each current sample
    nearest_dist = np.min(dist_matrix, axis=1)
    # Adding density considerations
    nearest_dist = nearest_dist * neigh_num

    weights = 1 / np.exp(nearest_dist)

    return weights
class scRNAMatrixInstance(Dataset):
    def __init__(self,
                 adata: AnnData = None,
                 obs_label_colname: str = "x",
                 transform: bool = False,
                 args_transformation_list: dict = {}
                 ):

        super().__init__()

        self.adata = adata
        adata = sc_tsne(adata, group="Group")
        if isinstance(self.adata.X, np.ndarray):
            self.data = self.adata.X
        else:
            self.data = self.adata.X.toarray()

        # label (if exist, build the label encoder)
        if self.adata.obs.get(obs_label_colname) is not None:
            self.label = self.adata.obs[obs_label_colname]
            self.unique_label = list(set(self.label))
            self.weights = get_weights(adata.obsm['X_tsne'])
            self.weights_by_density = get_weights_by_density(adata.obsm['X_tsne'])
            self.weights_by_distance = get_weights_by_distance(adata.obsm['X_tsne'])
            self.label_encoder = {k: v for k, v in zip(self.unique_label, range(len(self.unique_label)))}
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        else:
            self.label = None
            print("Can not find corresponding labels")

        # do the transformation
        self.transform = transform
        self.num_cells, self.num_genes = self.adata.shape
        self.args_transformation_list = args_transformation_list
        
        self.dataset_for_transform = deepcopy(self.data)
        
    def RandomTransform(self, sample, args_transformation):
        #tr_sample = deepcopy(sample)
        tr = transformation(self.dataset_for_transform, sample)
        # the crop operation
        # Mask
        tr.random_mask(args_transformation['mask_percentage'], args_transformation['apply_mask_prob'])
        # (Add) Gaussian noise
        tr.random_gaussian_noise(args_transformation['noise_percentage'], args_transformation['sigma'],
                                 args_transformation['apply_noise_prob'])
        tr.ToTensor()

        return tr.cell_profile


    def __getitem__(self, index):
        
        sample = self.data[index].astype(np.float32)

        if self.label is not None:
            label = self.label_encoder[self.label[index]]
        else:
            label = -1

        if self.transform:
            # 仅对其中一个样本做增强
            sample_a = torch.Tensor(sample)
            sample_p_1 = self.RandomTransform(sample, self.args_transformation_list[0]).float()
            sample_p_2 = self.RandomTransform(sample, self.args_transformation_list[1]).float()
            sample = [sample_a, sample_p_1, sample_p_2]

        else:
            sample_a = torch.Tensor(sample)
            sample = [sample_a]
        
        return sample, index, label

    def __len__(self):
        return self.adata.X.shape[0]


class transformation():
    def __init__(self, 
                 dataset,
                 cell_profile):
        self.dataset = dataset
        self.cell_profile = deepcopy(cell_profile)
        self.gene_num = len(self.cell_profile)
        self.cell_num = len(self.dataset)
    
    
    def build_mask(self, masked_percentage: float):
        mask = np.concatenate([np.ones(int(self.gene_num * masked_percentage), dtype=bool), 
                               np.zeros(self.gene_num - int(self.gene_num * masked_percentage), dtype=bool)])
        np.random.shuffle(mask)
        return mask



    def random_mask(self, 
                    mask_percentage: float = 0.15, 
                    apply_mask_prob: float = 0.5):

        s = np.random.uniform(0,1)
        if s<apply_mask_prob:
            # create the mask for mutation
            mask = self.build_mask(mask_percentage)
            
            # do the mutation with prob
            self.cell_profile[mask] = 0

    def random_gaussian_noise(self,
                              noise_percentage: float = 0.2,
                              sigma: float = 0.5,
                              apply_noise_prob: float = 0.3):

        s = np.random.uniform(0, 1)
        if s < apply_noise_prob:
            # create the mask for mutation
            mask = self.build_mask(noise_percentage)

            # create the noise
            noise = np.random.normal(0, sigma, int(self.gene_num * noise_percentage))

            # do the mutation (maybe not add, simply change the value?)
            self.cell_profile[mask] += noise

    def ToTensor(self):
        self.cell_profile = torch.from_numpy(self.cell_profile)