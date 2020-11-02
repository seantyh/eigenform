import pickle
import numpy as np
from .utils import get_data_dir
from sklearn.utils.extmath import randomized_svd

def make_svd(font_prefix):    
    svd_dir = get_data_dir() / "svd"
    svd_path = svd_dir / f"svd_{font_prefix}.pkl"
    
    if svd_path.exists():
        with svd_path.open("rb") as fin:
            U, S, Vt = pickle.load(fin)
        return (U, S, Vt)
    else:
        data_dir = get_data_dir()
        with (data_dir / f"char_img/char_img_{font_prefix}.pkl").open("rb") as fin:
            M = pickle.load(fin)
        U, S, Vt = randomized_svd(M, n_components=500, random_state=3423)

        with svd_path.open("wb") as fout:
            pickle.dump((U, S, Vt), fout)
            # print(f"dimensions: U({U.shape}), S({S.shape}), V({Vt.shape})")
        return (U, S, Vt)

class SvdUtils:
    def __init__(self, svd_path, k):
        with svd_path.open("rb") as fin:
            svd = pickle.load(fin)
        self.U = svd[0][:, :k]
        self.sigma = svd[1][:k]
        self.Vt = svd[2][:k, :]
    
    def project_svd(self, im_vec):
        sigma = self.sigma
        U = self.U
        A = np.dot(np.diag(1/sigma), np.transpose(U))
        proj = np.dot(A, im_vec)
        return proj

    def recon_svd(self, coeff_mat):
        sigma = self.sigma
        U = self.U
        A = np.dot(U, np.diag(sigma))
        recon_mat = np.dot(A, coeff_mat)
        return recon_mat
    