# Data Science Portfolio

# [Project 4: Customer Profiling](https://github.com/DomS1080/Data-Science/blob/main/Projects/Customer/Customer.ipynb)
   - In-progress
   - Apply various techniques (unsupervised, supervised) to reduce dataset dimensionality, predict customer segment, and analyze customer clusters
     * Dimensionality reduction using Principal Component Analysis (PCA)
     * Segmentation of ~950 customer records using ~5700 labelled instances 
     • Profile cluster characteristics, relationship of cluster to spend scores, and developing ideal customer profile
   - Metrics: Accuracy
   - Key Project Components:
     * Market Segmentation, Cluster Analysis, Dimensionality Reduction, Principal Component Analysis (PCA)
   - Libraries:
     * Scikit-learn
     * Matplotlib
     * Pandas
     * Numpy

# [Project 3: Computer Vision Neural Networks](https://github.com/DomS1080/Data-Science/blob/main/Projects/Deep%20Neural%20Network/Computer%20Vision/F-MNIST%20Tensorflow%20MLPs.ipynb)
   - Built, trained, and compared performance between Tensorflow Multilayer Perceptron (MLP) networks for a 10-class clothing image classification task
     * Compared training characteristics and performance between MLP structures using benchmarks set by a baseline model
     * Implemented cost-saving measures such as improved optimizers and early stopping based on loss monitoring
     * Visualized training with Matplotlib & Tensorboard
     * Evaluated each model's predictive performance on test set; overall and by label
   - Metrics: Accuracy, Sparse Categorical Crossentropy Loss
   - Key Project Components:
     * Artificial/Deep Neural Network, Computer Vision, Deep Learning, Multiclass Classification, Error Analysis
   - Libraries:
     * TensorFlow / Tensorboard
     * Scikit-learn
     * Matplotlib
     * Pandas
     * Numpy

# [Project 2: Gold Price Time-Series Forecasting](https://github.com/DomS1080/Data-Science/blob/main/Projects/Supervised%20Learning/Time%20Series%20Forecasting/Gold%20Darts%20N-BEATS%20BlockRNN.ipynb)
   - Section 1: Trained Darts model with N-BEATS architecture to predict closing prices for Gold (1 oz; 'GC=F') over 3 mo. period prior to/after specified date, in varying prediction increments (33, 66 days)
     * Evaluated predictions against ground truth values (for 'prior 3 mo.' predictions)
   - Section 2: Utilized Darts BlockRNN model architecture to reduce error by training with 3 selected covariates
   - Metric: Mean Average Percent Error (MAPE)
   - Key Project Components:
     * Deep Neural Network, Time Series Forecasting, Data API
   - Libraries:
     * Yfinance
     * Darts
     * Matplotlib
     * Pandas

# [Project 1: Boston Housing Prices Multiple Linear Regression](https://github.com/DomS1080/Data-Science/blob/main/Projects/Supervised%20Learning/Regression/Boston%20Multiple%20Linear%20Regression.ipynb)
   - Utilized Scikit-Learn's implementation of the Multiple Linear Regression algorithm to predict median housing values by neighborhood, based on neighborhood data/characteristics
     * Predictive performance and next steps for iterative analysis are discussed
   - Metric: Root Mean Squared Error (RMSE)
   - Key Project Components:
     * Data Exploration, Preprocessing, Feature Engineering
   - Libraries:
     * Scikit-learn
     * Statsmodels
     * Seaborn
     * Matplotlib
     * Pandas
     * Numpy
