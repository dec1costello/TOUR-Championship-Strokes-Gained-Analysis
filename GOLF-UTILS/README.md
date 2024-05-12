## **edaUtils Install**

### Navigate to edaUtils in the terminal to create the wheel file:

```console
python3 -m build
```

### Place the newly created wheel file in your desried directory and finish install in notebook:
```console
pip install golfutils-1.0-py2.py3-none-any.whl
```

### Use edaUtils!

```console
from golfutils import util_funcs
temp_df  = util_funcs.tempGeoDf(df,3)
```
