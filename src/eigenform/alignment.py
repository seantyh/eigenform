import numpy as np

def make_coeff(vectors, k, axis=0):    
    if axis == 0:
        vec_k = vectors[:k, :]
    elif axis == 1:
        vec_k = vectors[:, :k]
    else:
        raise ValueError("not supported axis: " + str(axis))

    vec_kc = vec_k - vec_k.mean(axis)
    vec_kcn = vec_kc / np.sqrt((vec_kc**2).sum(axis))
    return vec_kcn

def align_space(target_svd, ref_svd, n_components):
    ref_coeff = make_coeff(ref_svd[2], n_components)
    target_coeff = make_coeff(target_svd[2], n_components)

    m = target_coeff.dot(ref_coeff.T)
    u, _, v = np.linalg.svd(m)
    ortho = u.dot(v)

    target_Ua = target_svd[0][:,:n_components].dot(ortho.T)
    target_Sa = target_svd[1][:n_components]
    target_Vta = ortho.dot(target_svd[2][:n_components, :])

    align_svd = (target_Ua, target_Sa, target_Vta)
    supp = {"ortho": ortho}
    return align_svd, supp