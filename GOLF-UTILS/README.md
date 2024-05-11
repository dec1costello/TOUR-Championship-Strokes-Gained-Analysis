## **GOLF-UTIL INSTALL**

### Navigate to GOLF-UTILS in the terminal to create the wheel file:
- delete this README.md beforehand
```console
python3 -m pip install build
```

```console
python3 -m build
```

### Install and use in a local notebook:
```console
pip install golfutils-1.0-py2.py3-none-any.whl
```

```console
from golfutils import util_funcs
temp_df  = util_funcs.tempGeoDf(df,3)
```
