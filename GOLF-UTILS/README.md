## **GOLF-UTIL INSTALL**

### Navigate to GOLF-UTILS in the terminal, run these commands to create the wheel file:
```console
python3 -m pip install build
```

```console
python3 -m build
```

### In your ipynb file, install and use locallly with these commands:
```console
pip install golfutils-1.0-py2.py3-none-any.whl
```

```console
from golfutils import util_funcs
temp_df  = util_funcs.tempGeoDf(df,3)
```
