import os

def get_font_path(name):
    font_dict = {        
        "黑體": "DFFN_B5.ttc",
        "楷體": "DFFN_K5.ttc",
        "少女體": "DFFN_H5.TTC",
        "仿宋體": "DFFN.TTC", 
        "行書體": "DFFN_S5.TTC",
        "隸書體": "DFFN_L5.TTC"
    }

    font_path = None
    for font_name in font_dict.keys():
        if name in font_name:
            font_path = os.path.join(
                os.path.dirname(__file__),
                "../../../data/fonts",
                font_dict[font_name]
            )
            return os.path.abspath(font_path)
    
    if not font_path:
        raise ValueError("Cannot find font file of " + name)