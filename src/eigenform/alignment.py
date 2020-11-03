import numpy as np
from typing import Tuple
from .ec_space import make_coeff

SvdTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]

def align_space(
        target_svd: SvdTuple, 
        ref_svd: SvdTuple, 
        n_components: int):
    # k: number of components
    # |V|: vocabulary size
    # |S|: number of pixels in image, e.g. 64*64 = 4096

    # coeff: k x |V|
    ref_coeff = make_coeff(ref_svd[2], n_components)
    target_coeff = make_coeff(target_svd[2], n_components)

    # m: k x k
    m = target_coeff.dot(ref_coeff.T)
    u, _, v = np.linalg.svd(m)
    ortho = u.dot(v)

    # aligned_Ua: |S| x k
    aligned_Ua = target_svd[0][:,:n_components].dot(ortho.T)    

    # aligned_coeff: k x |V|
    aligned_coeff = ortho.dot(target_coeff)

    aligned_space = {
        "bases": aligned_Ua, 
        "coeff": aligned_coeff
    }

    supp = {"ortho": ortho}
    return aligned_space, supp