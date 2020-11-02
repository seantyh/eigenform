from argparse import ArgumentParser
from import_pkg import eigenform
import json
import pickle

def main(args):
    data_dir = eigenform.get_data_dir()
    with open(data_dir / "vocabs.txt", encoding="UTF-8") as fin:
        vocabs = json.load(fin)
        
    if args.vocab_size:
        vocabs = vocabs[:args.vocab_size]

    textstr = "".join(vocabs)
    font_path = eigenform.get_font_path(args.font_name)
    mat, _ = eigenform.text2matrix(textstr, 
            im_dim=(64*len(textstr), 64), font_path=font_path)

    out_path = data_dir / f"char_img_{args.prefix}.pkl"
    with out_path.open("wb") as fout:
        pickle.dump(mat, fout)
    print("character image matrix written to ", out_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--font-name",             
            choices=["黑體", "楷體", 
                     "少女體", "仿宋體", "行書體", "隸書體"])
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--vocab-size", default=None, type=int)
    args = parser.parse_args()
    main(args)
