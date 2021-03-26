# Interpretable Machine Learning with Python

<a href="https://www.packtpub.com/product/interpretable-machine-learning-with-python/9781800203907"><img src="https://static.packt-cdn.com/products/9781800203907/cover/smaller" alt="Interpretable Machine Learning with Pythone" height="256px" align="right"></a>

This is the code repository for [Interpretable Machine Learning with Python](https://www.packtpub.com/product/interpretable-machine-learning-with-python/9781800203907), published by Packt.

**Learn to build interpretable high-performance models with hands-on real-world examples**

## What is this book about?
Do you want to understand your models and mitigate the risks associated with poor predictions using practical machine learning (ML) interpretation? Interpretable Machine Learning with Python can help you overcome these challenges, using interpretation methods to build fairer and safer ML models.

This book covers the following exciting features: <First 5 What you'll learn points>
* Recognize the importance of interpretability in business
* Study models that are intrinsically interpretable such as linear models, decision trees, and Naïve Bayes
* Become well-versed in interpreting models with model-agnostic methods
* Visualize how an image classifier works and what it learns
* Understand how to mitigate the influence of bias in datasets

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/180020390X) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
base_classifier = KerasClassifier(model=base_model,\
                                  clip_values=(min_, max_))
y_test_mdsample_prob = np.max(y_test_prob[sampl_md_idxs],\
                                                       axis=1)
y_test_smsample_prob = np.max(y_test_prob[sampl_sm_idxs],\
                                                       axis=1)
```

**Following is what you need for this book:**
This book is for data scientists, machine learning developers, and data stewards who have an increasingly critical responsibility to explain how the AI systems they develop work, their impact on decision making, and how they identify and manage bias. Working knowledge of machine learning and the Python programming language is expected.

With the following software and hardware list you can run all code files present in the book (Chapter 1-14).

### Software and Hardware List

You can install the software required in any operating system by first installing [Jupyter Notebook or Jupyter Lab](https://jupyter.readthedocs.io/en/latest/install.html) with the most recent version of Python, or install [Anaconda](https://docs.anaconda.com/anaconda/) which can install everything at once. While hardware requirements for Jupyter are relatively modest, we recommend a machine with at least 4 cores of 2Ghz and 8Gb of RAM.

Alternatively, to installing the software locally, you can run the code in the cloud using Google Colab or another cloud notebook service.  

Either way, the following packages are required to run the code in all the chapters (Google Colab has all the packages denoted with a ^):

| Chapter  | Software required                   | OS required                        |
| -------- | ------------------------------------| -----------------------------------|
| 1  - 14      | ^ Jupyter Notebook / Lab                     | Windows, Mac OS X, and Linux (Any) |
| 1  - 14       | ^ Python 3.6+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14      | ^ numpy 1.19.5+           | Windows, Mac OS X, and Linux (Any) |
| 1  - 14  | ^ pandas 1.1.5+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14   | ^ scikit-learn 0.22.2+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14  | ^ matplotlib 3.2.2+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14   | ^ scipy 1.4.1+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14   | ^ beautifulsoup4 4.6.3+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14       | ^ requests 2.23.0+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14       | ^ statsmodels 0.10.2+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | ^ seaborn 0.11.1+           | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | ^ tqdm 4.41.1+           | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | pathlib2 2.3.5+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14       | ^ mlxtend 0.14.0+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | pycebox 0.0.1+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | alibi 0.5.5+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | aif360 0.3.0+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | ^ opencv-python 4.5.1+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | machine-learning-datasets 0.01.16+           | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | rulefit 0.3.1+           | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | interpret 0.2.2+           | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | skope-rules 1.0.1+           | Windows, Mac OS X, and Linux (Any) |
| 1  - 14        | ^ six 1.15.0+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14       | ^ tensorflow 2.4.1+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14       | cvae 0.0.3+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14       | PDPbox 0.2.0+            | Windows, Mac OS X, and Linux (Any) |
| 1  - 14      | pycebox 0.0.1+           | Windows, Mac OS X, and Linux (Any) |

sklearn-genetic                    0.3.0
                   

!pip install --upgrade xgboost tensorflow keras shap 
!pip install git+https://github.com/tensorflow/docs
!pip install git+https://github.com/MaximeJumelle/ALEPython.git@dev#egg=alepython
!pip install --upgrade nltk lightgbm lime
!pip install --upgrade catboost alibi witwidget
!pip install --upgrade opencv-python tf-explain tf-keras-vis scikit-image
!pip install --upgrade distython SALib
!pip install --upgrade yellowbrick mlxtend sklearn-genetic
!pip install numba==0.49 
!pip install --upgrade BlackBoxAuditing
!pip install --upgrade aif360  
!pip install --upgrade econml dowhy
!pip install --no-deps git+https://github.com/EthicalML/xai.git
!pip install --upgrade bayesian-optimization tensorflow-lattice graphviz pydot
!pip install --upgrade adversarial-robustness-toolbox

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781800203907_ColorImages.pdf).

### Related products <Other books you may enjoy>
* Automated Machine Learning [[Packt]](https://www.packtpub.com/product/automated-machine-learning/9781800567689) [[Amazon]](https://www.amazon.com/dp/1800567685)

* Hands-On Machine Learning with scikit-learn and Scientific Python Toolkits [[Packt]](https://www.packtpub.com/product/hands-on-machine-learning-with-scikit-learn-and-scientific-python-toolkits/9781838826048) [[Amazon]](https://www.amazon.com/dp/1838826041)

## Get to Know the Authors
**Serg Masís**
has been at the confluence of the internet, application development, and analytics for the last two decades. Currently, he's a Climate and Agronomic Data Scientist at Syngenta, a leading agribusiness company with a mission to improve global food security. Before that role, he co-founded a startup, incubated by Harvard Innovation Labs, that combined the power of cloud computing and machine learning with principles in decision-making science to expose users to new places and events. Whether it pertains to leisure activities, plant diseases, or customer lifetime value, Serg is passionate about providing the often-missing link between data and decision-making — and machine learning interpretation helps bridge this gap more robustly.
