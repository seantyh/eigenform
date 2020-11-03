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

class FormSpace:
    def __init__(self, label, vocabs, n_components, bases, coeff):
        self.__n_components = n_components
        self.label = label
        self.vocabs = vocabs        
        self.cv_dist, self.vectors = self.build_index(coeff, n_components)
        self.stoi = {x: i for i, x in enumerate(vocabs)}
        self.itos = {i: x for i, x in enumerate(vocabs)}

        if bases:
            self.raw_vectors = coeff.T
            self.bases = bases
        else:
            self.raw_vectors = None
            self.bases = None
        

    def __repr__(self):
        return f"<FormSpace: {self.label}, {len(self.vocabs)} x {self.n_components}>"

    @property
    def n_components(self):
        return self.__n_components

    @staticmethod
    def from_svd(label, vocabs, n_components, svd_tuple):
        U, S, Vt = svd_tuple
        # coeff: k x |V|
        coeff = make_coeff(Vt, n_components)

        # bases: |S| x k
        bases = U[:,:n_components]
        space = FormSpace(label, vocabs, n_components, bases, coeff)

    def build_index(self, coeff_vectors, k):
        # coeff_vectors is assume to be k x |V|
        nr, nc = coeff_vectors.shape
        if nr > nc:
            print(f"WARNING: coefficient size is {nr} x {nc}")
            print("it is assumed to be k x |V|")
        
        coeff_mat = coeff_vectors.T[:, :k]
        vec_kc = coeff_mat - coeff_mat.mean(1)[:, np.newaxis]
        vec_kcn = vec_kc / np.sqrt((vec_kc**2).sum(1))[:, np.newaxis]
        
        cv_dist = np.dot(vec_kcn, vec_kcn.T)
        return cv_dist, vec_kcn

    def most_similar(self, char, nMax=20):
        cv_dist = self.cv_dist
        stoi = self.stoi
        itos = self.itos
        neighbors = np.argsort(-cv_dist[stoi[char], :])[:nMax]  #pylint: disable=invalid-sequence-index
        similars = [itos[ch_i] for ch_i  in neighbors]
        return similars

    def most_loaded(self, compo_i, ascending=False):
        compo_vec = self.raw_vectors[:, compo_i]
        if ascending:
            sort_idx = np.argsort(compo_vec)
        else:
            sort_idx = np.argsort(-compo_vec)
        return [self.itos[i] for i in sort_idx[:10]]

    def get_vector(self, char):
        if char not in self.stoi:
            return None
        chidx = self.stoi[char]
        return self.vectors[chidx, :]

    def recon(self, char=None, chidx=None):
        if not char and not chidx:
            raise ValueError("One of char and chidx must be not None")

        if char:
            chidx = self.xtoi[char]
        
    def neighbors(self, char, threshold=0.34):
        ## the default threshold is determined with
        ## the 99.5 quantile of pairwise similarities

        cv_dist = self.cv_dist
        stoi = self.stoi
        itos = self.itos
        dists = cv_dist[stoi[char], :]
        sort_idx = np.argsort(-dists)
        neigh_chars = [itos[chidx] for chidx
                       in sort_idx 
                       if dists[chidx] > threshold]
        return neigh_chars
    
    def neighbors_vector(self, invec, threshold=0.34):
        dists = self.vectors.dot(invec.T).squeeze()
        itos = self.itos
        sort_idx = np.argsort(-dists)
        neighs = [itos[chidx] for chidx
                       in sort_idx 
                       if dists[chidx] > threshold]
        return neighs
    
    def neighbors_sim_vector(self, invec, n_neigh=20):
        dists = self.vectors.dot(invec.T).squeeze()
        itos = self.itos
        sort_idx = np.argsort(-dists)[:n_neigh]
        neigh_sims = np.array([dists[chidx] for chidx
                       in sort_idx])
        return neigh_sims

    def neighbors_sim(self, char, n_neigh=20):
        stoi = self.stoi
        itos = self.itos
        dists = self.cv_dist[stoi[char]]        
        sort_idx = np.argsort(-dists)[:n_neigh]
        neigh_sims = np.array([dists[chidx] for chidx
                       in sort_idx])
        return neigh_sims

        


