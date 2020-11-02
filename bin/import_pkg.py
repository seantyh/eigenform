import sys, os
pkg_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../src"))
if pkg_path not in sys.path:
    sys.path.append(pkg_path)

import eigenform