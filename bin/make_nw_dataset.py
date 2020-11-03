from argparse import ArgumentParser
import pickle
import numpy as np
from import_pkg import eigenform
from tqdm.auto import tqdm
import pandas as pd

def main(args): 
    norm_space_path = eigenform.get_data_dir()/"norm/norm_space.pkl"
    print("Loading normalized character space")
    with norm_space_path.open("rb") as fin:
        nspace = pickle.load(fin)   

    print("Loading word space")
    w2_space_path = eigenform.get_data_dir()/"norm/word2_space.pkl"
    with w2_space_path.open("rb") as fin:
        w2_space = pickle.load(fin)    
        w2_space.cv_dist = None

    with (eigenform.get_data_dir()/"asbc5_characters.pkl").open("rb") as fin:
        cfreq = pickle.load(fin)    
    
    
    non_word_ldt = pd.read_excel(eigenform.get_data_dir()/"Tse-2017-Chinese-lexicon-project.xlsx", sheet_name="Nonword")
    # non_word_ldt = non_word_ldt.iloc[:10, :]

    c1_freqs = []
    c2_freqs = []
    c1_neigh_counts = []
    c2_neigh_counts = []
    nw_neigh_counts = []
    for nw in tqdm(non_word_ldt.Nonword_Trad.tolist()):    
        c1 = nw[0]; c2 = nw[1]
        try:
            c1_freq = cfreq.get(c1, 1)
            c2_freq = cfreq.get(c2, 1)
            c1_nNeigh = len(nspace.neighbors(c1, threshold=0.34))
            c2_nNeigh = len(nspace.neighbors(c2, threshold=0.34))
            nw_vec = np.concatenate([nspace.get_vector(nw[0]), nspace.get_vector(nw[1])], axis=0)    
            nw_nNeigh = len(w2_space.neighbors_vector(nw_vec, 0.28))
        except:
            c1_freq = 1; c2_freq = 1
            c1_nNeigh = None; c2_nNeigh = None
            nw_nNeigh = None
        c1_freqs.append(c1_freq)
        c2_freqs.append(c2_freq)
        c1_neigh_counts.append(c1_nNeigh)
        c2_neigh_counts.append(c2_nNeigh)
        nw_neigh_counts.append(nw_nNeigh)
    
    space_nw = non_word_ldt.assign(
        c1_logfreq=np.log(c1_freqs), c2_logfreq=np.log(c2_freqs),
        c1_nNeigh=c1_neigh_counts, c2_nNeigh=c2_neigh_counts,
        nw_nNeigh=nw_neigh_counts
    )    
    space_nw.dropna(inplace=True)    

    space_nw.to_csv(eigenform.get_data_dir()/"norm/norm_space_ldt.csv")
    print("dataset saved")

if __name__ == "__main__":
    parser = ArgumentParser()    
    args = parser.parse_args()
    main(args)