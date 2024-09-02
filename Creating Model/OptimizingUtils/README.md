<!-- STRUCTURE -->
<h2 id="Util-Structure"> 🌵 Util Overview</h2>

The contents of this package is organized as follows:

    📂 OptmizingUtils 📍
    │
    ├── README.md
    ├── pyproject.toml
    │   
    └── 📂 OptmizingUtils/tunePipeline
        ├── tunePipeline.py 
        └── __init__.py 

<h2 id="Install-Instructions"> ⬇️ Install Instructions</h2>

Navigate to OptmizingUtils in the terminal and create the wheel file:

```console
python3 -m build
```

Place the newly created wheel file in desried directory and finish install in notebook:

```console
pip install optimizingutils-1.85-py2.py3-none-any.whl
```

<h2 id="Preprocessing"> ⚙️ Preprocessing</h2>

#### Scalers

* Standard
* Minmax
* Maxabs
* Robust
* Quantile
* Power

----------------------------------------------------------------

#### Encoders

* OneHot
* Ordinal
* Target
* Binary
* Hashing
* Helmert

----------------------------------------------------------------

#### Imputers

* SimpleImputer
    * Mean
    * Median
    * Most Frequent
    * Constant
* KNNImputer
* IterativeImputer
  
----------------------------------------------------------------

#### Feature Selectors

* None
* Kbest
    * F-regression
    * Mutual Info Classif
* Model
    * Gradient Boosting Regressor
    * Random Forest Classifier
* RFE
    * Gradient Boosting Regressor
    * Random Forest Classifier
* RFECV
    * Gradient Boosting Regressor
    * Random Forest Classifier
* Polynomial Features
    * Poly Degree
    * Include Bias
