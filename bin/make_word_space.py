from argparse import ArgumentParser
import pickle
import re
import numpy as np
from import_pkg import eigenform
from tqdm.auto import tqdm

def main(args): 
    norm_space_path = eigenform.get_data_dir()/"norm/norm_space.pkl"
    print("Loading normalized character space")
    with norm_space_path.open("rb") as fin:
        nspace = pickle.load(fin)   

    print("Loading ASBC5 word lexicon")
    with (eigenform.get_data_dir()/"asbc5_words.pkl").open("rb") as fin:
        asbc_words = pickle.load(fin)

    print("Finding 2-char words and freq > 10: ", end="")
    words = []
    for w, f in asbc_words.items():
        if f <= 20 or len(w)!=2:
            continue
        if not re.match("[\u4e00-\ufeff]+", w):
            continue
        words.append(w)
    print(len(words))

    print("Building word space")
    coeffs = []
    word_vocab = []
    for w in tqdm(words):
        c1_vec = nspace.get_vector(w[0])
        c2_vec = nspace.get_vector(w[1])
        if c1_vec is None or c2_vec is None:
            continue
        cat_vec = np.concatenate([c1_vec, c2_vec], axis=0)
        coeffs.append(cat_vec)    
        word_vocab.append(w)
    coeffs = np.stack(coeffs)
    w2_space = eigenform.FormSpace("w2", word_vocab, 200, None, coeffs.T)

    # Saving    
    print("Saving word space...", end="")
    word_space_path = eigenform.get_data_dir()/"norm/word2_space.pkl"
    with word_space_path.open("wb") as fout:
        pickle.dump(w2_space, fout)
    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser()    
    args = parser.parse_args()
    main(args)