from argparse import ArgumentParser
from import_pkg import eigenform
import json
from tqdm.auto import tqdm
from itertools import chain
from matplotlib import pyplot as plt
import pickle

def plot_ECs_50(U, fig_path):
    fig, axes = plt.subplots(5, 10, squeeze=False)
    fig.set_size_inches(12, 8)
    for ec_i, ax in enumerate(chain.from_iterable(axes)):        
        ax.imshow(eigenform.recon(U[:,ec_i], 64), cmap='gray')
        ax.axis('off')
        ax.set_title(f'EC{ec_i}')
    fig.savefig(fig_path)

def main(args):
    ref_font = "hei"
    font_list = ["hei", "kai", "shao", "li", "xing"]
    hei_svd = eigenform.make_svd("hei")

    ec_path = eigenform.get_data_dir() / f"svd/ec50_hei.png"
    plot_ECs_50(hei_svd[0], ec_path)
    for font_x in tqdm(font_list):
        svd_x = eigenform.make_svd(font_x)
        align_space_x, supp = eigenform.align_space(svd_x, hei_svd, args.n_components)
        out_svd_path = eigenform.get_data_dir() / f"svd/align_space_{font_x}.pkl"
        ec_path = eigenform.get_data_dir() / f"svd/ec50_{font_x}.png"
        align_ec_path = eigenform.get_data_dir() / f"svd/align_ec50_{font_x}.png"
        with out_svd_path.open("wb") as fout:
            pickle.dump({
                "align_bases": align_space_x["bases"],
                "align_coeff": align_space_x["coeff"],
                "align_mat": supp["ortho"]
            }, fout)
        plot_ECs_50(svd_x[0], ec_path)
        plot_ECs_50(align_space_x["bases"], align_ec_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-components", default=100, type=int)
    args = parser.parse_args()
    main(args)