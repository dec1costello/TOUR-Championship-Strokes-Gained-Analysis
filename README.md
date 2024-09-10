[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dec1costello/TOUR-Championship-Strokes-Gained-Analysis/HEAD)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dec1costello/TOUR-Championship-Strokes-Gained-Analysis)
<br />
*Author: Declan Costello*

<a name="readme-top"></a>

<p align="center">
<img height="150" width="800" src="https://github.com/dec1costello/Golf/assets/79241861/f6f27eb6-5943-4e56-88b1-303248913ed1"/>  
</p>

<h1 align="center">TOUR Championship Analysis</h1>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li><a href="#Repo-Overview">Repo Overview</a></li>
    <li><a href="#Code-Quality">Code Quality</a></li>
    <li><a href="#Dataset">Dataset</a></li>
    <li><a href="#EDA">EDA</a></li>
    <ol>
    <li><a href="#SG-per-Round">SG per Round</a></li>
    <li><a href="#SG-per-Hole">SG per Hole</a></li>
    <li><a href="#SG-per-Drive">SG per Drive</a></li>
    </ol>
    <li><a href="#Expected-Strokes-Model">Expected Strokes Model</a></li>
    <ol>
    <li><a href="#Model-Selection">Model Selection</a></li>
    <li><a href="#Training-Architecture">Training Architecture</a></li>
    <li><a href="#Fighting Bias">Fighting Bias</a></li>
    <li><a href="#Model-Performance">Model Performance</a></li>
    </ol>
    <li><a href="#Applying-xS-Model">Applying xS Model</a></li>
    <ol>
    <li><a href="#SG-per-Shot-Type">SG per Shot Type</a></li>
    </ol>
    <li><a href="#Conclusion">Conclusion</a></li>
    <li><a href="#Future-Roadmap">Future Roadmap</a></li>
  </ol>
</details>

## **🎯 Objectives**

Welcome to my analysis of the 2011 TOUR Championship at East Lake Golf Club, the primary objective of this project is to:
> **Develop an expected strokes model to identify player performance**

I hope to contribute meaningful insights to the golf community through this project. Although the 2011 TOUR Championship took place over a decade ago and the tournament's rules have since changed, its extensive shot-level dataset remains a valuable resource. If you happen to come across another complete shot-level dataset, I would greatly appreciate it if you could share it with me! I encourage you to check out the js visuals on [NBViewer!](https://nbviewer.org/github/dec1costello/TOUR-Championship-Strokes-Gained-Analysis/tree/main/)

## **🌵 Repo Structure**

This repo is organized as follows:

    📂 TOUR-Championship-Strokes-Gained-Analysis 📍
    │
    ├── 📂 Data
    ├── CITATION
    ├── README.md
    ├── CODE_OF_CONDUCT.md
    │
    ├── 📂 EDA
    │   ├── EDA.ipynb
    │   ├── SGperHole.ipynb
    │   ├── SGperRound.ipynb
    │   ├── SGperDrive.ipynb
    │   ├── FeatureEngineering.ipynb
    │   └── 📂 EDAUtils
    │
    ├── 📂 Creating Model
    │   ├── LazyPredict.ipynb 
    │   ├── PuttingModel.ipynb
    │   ├── ApproachModel.ipynb 
    │   └── 📂 OptimizingUtils
    │
    ├── 📂 Applying Model
    │   ├── SGCreation.ipynb 
    │   └── SGAnalysis.ipynb
    │
    └── 📂 Streamlit Dashboard

## **⭐ Code Quality**

In this project, a Security Linter, Code Formatting, Type Checking, and Code Linting are essential for ensuring code quality and robustness. These help identify and mitigate security vulnerabilities, maintain consistent coding styles, enforce type safety, and detect potential errors or issues early in the development process, ultimately enhancing the reliability and maintainability of the project.

<div align="center">

| Security Linter | Code Formatting | Type Checking | Code Linting |
| ------------------------------------------- | -------------------------------------------------- | ---------------------------------------- | ------------------------------------------- |
| [`bandit`](https://github.com/PyCQA/bandit) | [`ruff-format`](https://github.com/astral-sh/ruff) | [`mypy`](https://github.com/python/mypy) | [`ruff`](https://github.com/astral-sh/ruff) |

</div>

## **📊 Dataset**

This dataset consists of shot level data from the PGA TOUR Championship. The TOUR Championship differs from other tournaments in that only the top 30 golfers compete and there's no cut after the second round, this ensures consistent data of high skill golfers across all 4 rounds. Additionally, it's important to acknowledge that the dataset lacks [data from the playoff that occurred](https://www.youtube.com/watch?v=vRjNR1T81VE), which is crucial for understanding the tournament's conclusion. Furthermore, it is important to emphasize that landing in the rough at East Lake doesn't necessarily disadvantage a player. Despite the challenge it presents, the ball could still have a favorable lie, which might have been strategically chosen by the golfer.

## 🔍 EDA

I analyze the data, focusing on feature engineering to understand, clean, and refine the dataset. This process guides model selection and validates assumptions, while also uncovering insights through visualization. By addressing data quality and recognizing patterns early on, I establish a solid foundation for the project. For instance, exploring Strokes Gained (SG) at the round, hole, and drive levels helps us make assumptions for building a model to examine SG on a shot-level basis later.

<img align="left" alt="scipi" width="32px" style="padding-right:3px;" src="https://github.com/dec1costello/Golf/assets/79241861/8c1b62d0-b4cb-46ba-82f0-a858508911ae" />
<img align="left" alt="bokeh" width="34px" style="padding-right:1px;" src="https://github.com/dec1costello/dec1costello/assets/79241861/bfbeaf3f-663e-4191-9e90-a70c322b0bd8" />

<br />
<br />

### SG per Round

I analyze the Strokes Gained distribution for each round of the Championship, revealing player performance trends during the tournament. This examination on a round-by-round basis helps uncover patterns in golfers' strategies and identifies challenges posed by difficult pin locations on the course.

#### Key Insights

* All rounds have a promising mean of 0
* Round 3 seemed to be the most chaotic, as there was a significant variance in player performance throughout the day

<div align="center">
    <img src="https://github.com/dec1costello/Golf/assets/79241861/275e7705-7748-49e2-b9c1-e7b24d40066d" alt="Event Scatter" style="width80%">
</div>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### SG per Hole

In this analysis, I investigate the distribution of Strokes Gained for each hole of every round of the Championship. Notably, Mahan ties Haas in Strokes Gained on the 72nd hole, a significant moment in the tournament. However, [Haas ultimately secured victory in the playoffs!](https://www.espn.com/golf/leaderboard?tournamentId=917)

#### Key Insights

* Players appear to continue to play relative to their initial performance of round 1
* Poorly performing players seem to give up come the back 9 of round 3

<div align="center">
    <img src="https://github.com/dec1costello/Golf/assets/79241861/5fb76665-1de7-4d00-a42d-370c6fc5a987" alt="Event Scatter" style="width:100%">
</div>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### SG per Drive

Here I explore the distribution of Strokes Gained vs Driving Distance Gained and Driving Accuracy Gained for each drive of the Championship. Both Driving Distance and Driving Accuracy are normalized per hole before totalling. Happy to say my analysis aligns with [Data Golf's Course Fit Tool.](https://datagolf.com/course-fit-tool)

#### Key Insights

* Driving Accuracy has a strong correlation to Strokes Gained per Hole
* Driving Distance has only a slight correlation to Strokes Gained per Hole

<div align="center">
    <img src="https://github.com/dec1costello/Golf/assets/79241861/2eabd308-cee6-4f72-af2b-6dcea8e6bd86" alt="Event Scatter" style="width:100%">
</div>
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## ⛳ Expected Strokes Model

The Stacked Expected Strokes Model leverages the power of ensemble learning by combining predictions from multiple base models to enhance accuracy and robustness. Notably, I've developed separate models for putting and approach scenarios, utilizing different input features tailored to each situation. This approach allows for more precise predictions by optimizing the model's focus on specific aspects of gameplay, ultimately leading to improved performance and insights in golf analytics. Furthermore, this model will eventually enable a granular analysis of shot-by-shot Strokes Gained, a significant departure from previous hole-by-hole and round-by-round evaluations. By harnessing the Stacked Expected Strokes Model's predictive capabilities, I'll unlock the ability to evaluate each shot's impact on overall performance, offering unprecedented insights into golfer performance. Additionally, I'm unconcerned about data leakage since I'll be predicting continuous variables while training on discrete data, ensuring the model's integrity and effectiveness in real-world applications.

<img align="left" alt="mlflow" width="28px" style="padding-right:3px;" src="https://github.com/dec1costello/dec1costello/assets/79241861/a59fbc5a-f5ce-47a9-a903-c4e5bb0e2e65" />
<img align="left" alt="optuna" width="33px" style="padding-right:3px;" src="https://github.com/dec1costello/dec1costello/assets/79241861/3a709d6c-cd1e-4126-bd83-ff0f958f4609" />
<img align="left" alt="scikit_learn" width="55px" style="padding-right:3px;" src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" />

<br />
<br />

### Model Selection

While the training data is discrete, for continuous predictions, I faced the task of selecting between regression models. As with all my models, I was sure to stratify the training and testing data before predicting. Initially, I employed [Lazy Predict](https://lazypredict.readthedocs.io/en/latest/) to assess various model options comprehensively.

#### Key Insight

* The GradientBoostingRegressor and HistGradientBoostingRegressor models performed the best
* If I were to have to constantly retrain the model I would avoid the MLPRegressor as it takes forever

<div align="center">  

| Model  | Adjusted R-Squared | R-Squared	| RMSE | Time Taken |
|-----------------------------------|-------|--------|-------|-------|
| GradientBoostingRegressor         | 0.85  | 0.85   | 0.46  | 0.93  |
| HistGradientBoostingRegressor     | 0.85  | 0.85   | 0.46  | 0.60  |
| LGBMRegressor                     | 0.85  | 0.85   | 0.47  | 0.14  |
| MLPRegressor                      | 0.84  | 0.84   | 0.48  | 5.23  |
| KNeighborsRegressor               | 0.82  | 0.83   | 0.50  | 0.16  |
| AdaBoostRegressor                 | 0.82  | 0.83   | 0.50  | 0.49  |
| RandomForestRegressor             | 0.82  | 0.82   | 0.50  | 3.46  |
| XGBRegressor                      | 0.82  | 0.82   | 0.50  | 0.24  |
| BaggingRegressor                  | 0.81  | 0.81   | 0.52  | 0.37  |
| NuSVR                             | 0.81  | 0.81   | 0.52  | 3.58  |
| ExtraTreesRegressor	              | 0.80  | 0.80   | 0.53  | 2.02  |
| SVR                               | 0.80  | 0.80   | 0.53  | 3.35  |

</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Training Architecture

After finding the top performing models, I ensemble the best models together using a [Stack](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html). In this project, I leveraged [Optuna's](https://optuna.org/#dashboard) CMAES Sampler to not only find the best parameters for each model in the stack resulting in minimized MAE, but also [data preprocessing scalers, encoders, imputation, and feature selection methods](https://github.com/dec1costello/TOUR-Championship-Strokes-Gained-Analysis/tree/main/Creating%20Model/OptimizingUtils). All trails are fed with appropriate offline training data from a [Feast](https://feast.dev/) feature store. I utilized a [ML Flow](https://medium.com/infer-qwak/building-an-end-to-end-mlops-pipeline-with-open-source-tools-d8bacbf4184f) model registry to track all Optuna trials. Databricks is leveraged to store production ready models. Finally, I wrapped this whole tuning process in a Poetry wheel file called 'OptimizingUtils' for reproducibility.

<div align="center">
    <img src="https://github.com/user-attachments/assets/c4b0cbb0-290d-4a3a-8572-779a810cc1ed" alt="Event Scatter" style="width:88%">
</div>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Fighting Bias

I attempted to prevent [Bias](https://towardsdatascience.com/a-quickstart-guide-to-uprooting-model-bias-f4465c8e84bc) by stratifying my training data and by using nested cross stratified split validation to prune biased trials. I plan to go a step further by bootstrapping, implementing [imbalanced learning libraries](https://www.kaggle.com/code/residentmario/undersampling-and-oversampling-imbalanced-data/notebook), and exploring Optuna's [terminator](https://optuna.readthedocs.io/en/stable/reference/terminator.html), [distribution](https://optuna.readthedocs.io/en/stable/reference/distributions.html), and [MultiObjectiveStudy](https://optuna.readthedocs.io/en/v2.0.0/reference/multi_objective/generated/optuna.multi_objective.study.MultiObjectiveStudy.html) feautres. I evaluate model bias that still occurred with [SHap](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) and [Lime](https://github.com/marcotcr/lime), enriching our understanding of the model's predictive behavior. Below, you'll find a SHap chart for the putting model's LGBMRegressor.

#### Key Insight
* Super surprised to see "Distance to Edge" matters more than "Distance to Pin" for putting, curious if this would be the case if I had a larger dataset
* "Downhill Slope" and "Elevation Below Ball" are distinct features; Despite their seemingly similar title, they are not the same. To confirm this, a pairwise correlation was done

<div align="center">
    <img src="https://github.com/dec1costello/Golf/assets/79241861/06baf5fd-bce3-4135-abe3-d9ba3b178d33" alt="Event Scatter" style="width:100%">
</div>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Model Performance

This chart helps evaluate the model by showing how predicted values compare to actual ones and revealing patterns in prediction errors. The histogram below assesses if errors follow a normal distribution, crucial for reliable predictions.

#### Key Insight

* Excited to see the residuals have a low standard deviation with a mean hovering around 0

<div align="center">
    <img src="https://github.com/dec1costello/Golf/assets/79241861/7b95ecab-a449-4770-a57e-eea884f1468b" alt="Event Scatter" style="width:100%">
</div>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🏌🏻 Applying xS Model

Now that we have a stacked SG machine learning model for a shot per shot basis, we can use it to gain valuable insights into golfer performance. Utilizing the model post-training enables golf analysts, coaches, and players to extract actionable insights, optimize strategies, and refine skills. Ultimately, leveraging a model empowers stakeholders to make informed decisions, enhance performance, and drive success on the golf course.

### SG per Shot Type

Now that we have a reliable model, we can use it to identify a player's strengths and weaknesses by subtracting Expected Strokes (xS) from the result of each shot to give us true Strokes Gained (SG). The plots below display Woodlands's Total SG and SG Percentile by shot type, providing a clear visualization of his performance across different lies and distances.

#### Key Insight

* Woodland was very successful gaining strokes on the green

<div align="center">
    <img src="https://github.com/user-attachments/assets/4d4d9911-3b2e-409f-ac45-63d19b0f3d13" alt="Event Scatter" style="width:100%">
</div>
<br />

* By looking at Woodland's SG Percentile, we can see that he truly underperformed from 200+ yards out, opposed to having one or two shots damage his 200+ SG Total
* Woodland only had six shots within 100-50 yards of the pin, perhaps this was by design to avoid putting himself in a position where he consistently underperforms


<div align="center">
    <img src="https://github.com/user-attachments/assets/f90faaed-113d-407b-97fc-bc41fcfffb58" alt="Event Scatter" style="width:100%">
</div>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🎬 Conclusion

Looking back, I wish I had known about Strokes Gained during my time as a caddy. I've come to understand that Strokes Gained provides a more accurate reflection of performance on the hole, while SG Percentiles based on shot location offer deeper insights into a golfer's true abilities. I'm excited to explore more golf-related projects in the future.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## 🗺️ Future Roadmap

- [ ] Model Refinement
    - [ ] [CI Orchestration](https://cml.dev/)
    - [x] [Model Registry](https://community.cloud.databricks.com/?o=5379878152283690)
    - [ ] [Drift Detection](https://www.youtube.com/watch?v=L4Pv6ExBQPM)
    - [ ] [Feature Store](https://feast.dev/) 
    - [ ] [Deploy](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-python-sdk/scikit_learn_randomforest/Sklearn_on_SageMaker_end2end.html)
- [ ] External Data
    - [ ] Player Course History
    - [ ] [Career Earnings](https://www.spotrac.com/pga/rankings/date/2011-01-01/2011-09-20/)
    - [ ] Equipment
    - [ ] Biometrics
    - [ ] [Weather](https://www.wunderground.com/history/daily/KATL/date/2011-9-22)
    - [ ] [SVGs](https://pgatour.bluegolf.com/bluegolfw/pgatour22/event/pgatour2210/course/eastlakegc/overview.htm)
    - [x] [HCP](https://pgatour.bluegolf.com/bluegolfw/pgatour22/event/pgatour2210/course/eastlakegc/detailedscorecard.htm)
- [ ] Bayesian Integration
    - [x] [Refer To](https://colab.research.google.com/github/AllenDowney/ThinkBayes2/blob/master/examples/hockey.ipynb#scrollTo=B-c6bb9wO-Cs)
    - [x] [Watch](https://www.youtube.com/watch?v=Zi6N3GLUJmw)
    - [ ] [Utilize](https://colab.research.google.com/github/AllenDowney/ThinkBayes2/)
<p align="center">
<img height="63" width="139" src="https://github.com/dec1costello/Golf/assets/79241861/0f9673d0-36c6-4d6f-928b-34d171a19350"/>
</p>
<p align="center">
<img height="375" width="100%" src="https://github.com/dec1costello/Golf/assets/79241861/506e2aa2-64e9-4383-83d9-d6e81f4dd5f7"/>
</p>
<p align="right">(<a href="#readme-top">back to top</a>)</p>
