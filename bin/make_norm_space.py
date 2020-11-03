from argparse import ArgumentParser
import json
import pickle
import json
from import_pkg import eigenform
from tqdm.auto import tqdm

def load_aligned_space(prefix):
    aligned_path = eigenform.get_data_dir() / f"svd/align_space_{prefix}.pkl"
    with aligned_path.open("rb") as fin:
        aspace = pickle.load(fin)
    return aspace

def main(args):
    hei_space = load_aligned_space("hei")
    norm_bases = hei_space["align_bases"]
    norm_coeff = hei_space["align_coeff"]
    font_list = "kai,li,shao,xing".split(",")
    for prefix in tqdm(font_list):
        aspace = load_aligned_space(prefix)
        norm_bases += aspace["align_bases"]
        norm_coeff += aspace["align_coeff"]
    norm_bases /= len(font_list)+1
    norm_coeff /= len(font_list)+1
    
    with (eigenform.get_data_dir()/"vocabs.json").open("r", encoding="UTF-8") as fin:
        vocabs = json.load(fin)
    
    eigenform.ensure_dir(eigenform.get_data_dir()/"norm")

    print("Normalizing coefficients...", end="")
    norm_coeff_path = eigenform.get_data_dir()/"norm/norm_space_coeff.pkl"
    norm_space_path = eigenform.get_data_dir()/"norm/norm_space.pkl"
    with norm_coeff_path.open("wb") as fout:
        pickle.dump({
            "align_bases": norm_bases,
            "align_coeff": norm_coeff            
        }, fout)
    print("Saved")

    print("Creating normalized FormSpace...", end="")
    norm_space = eigenform.FormSpace("norm", vocabs, args.n_components, norm_bases, norm_coeff)
    with norm_space_path.open("wb") as fout:
        pickle.dump(norm_space, fout)
    print("Saved")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-components", default=100, type=int)
    args = parser.parse_args()
    main(args)