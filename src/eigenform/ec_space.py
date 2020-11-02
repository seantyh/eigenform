import numpy as np

class FormSpace:
    def __init__(self, label, n_components=100, svd_tuple=None, vocabs=None):
        self.n_components = n_components
        self.label = label
        if svd_tuple and vocabs:
            U, S, Vt = svd_tuple 
            self.ec_bases = U[:,:n_components].dot(np.diag(S[:n_components]))            
            coeff_mat = np.dot(np.diag(S), Vt).T                        
            self.cv_dist, self.vectors = self.build_index(coeff_mat, n_components)
            self.raw_vectors = coeff_mat
            self.vocabs = vocabs
            self.stoi = {x: i for i, x in enumerate(vocabs)}
            self.itos = {i: x for i, x in enumerate(vocabs)}            
    
    def __repr__(self):
        return f"<FormSpace: {self.label}, {len(self.vocabs)} x {self.n_components}>"

    def get_ec_dim(self):
        return self.n_components

    def build_index(self, cv_vectors, k):
        cv_k = cv_vectors[:,:k]
        cv_kc = cv_k - cv_k.mean(1)[:, np.newaxis]
        cv_kcn = cv_kc / np.sqrt(np.diag(np.dot(cv_kc, cv_kc.transpose())))[:, np.newaxis]        
        cv_dist = np.dot(cv_kcn, cv_kcn.transpose())
        return cv_dist, cv_kcn
    
    def most_similar(self, char):
        cv_dist = self.cv_dist
        stoi = self.stoi
        itos = self.itos
        neighbors = np.argsort(-cv_dist[stoi[char], :])[:20]  #pylint: disable=invalid-sequence-index
        similars = [itos[ch_i] for ch_i  in neighbors]
        return similars
    
    def most_loaded(self, compo_i, ascending=False):
        compo_vec = self.raw_vectors[:, compo_i]
        if ascending:
            sort_idx = np.argsort(compo_vec)
        else:
            sort_idx = np.argsort(-compo_vec)
        return [self.itos[i] for i in sort_idx[:10]]                

    def get_ec_coeffs(self, char):
        if char not in self.stoi:
            return None
        chidx = self.stoi[char]
        return self.vectors[chidx, :]
    
    def recon(self, char=None, chidx=None):
        if not char and not chidx:
            raise ValueError("One of char and chidx must be not None")

        if char:
            chidx = self.xtoi[char]


