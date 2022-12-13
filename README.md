```python
import DaveML
from DaveML import DataLoader
from DaveML import LinearRegression
from DaveML import RidgeRegression
from DaveML import LogisticRegression
from pprint import pprint
```


```python
import numpy as np
```


```python
# Params: Filepath, delimiter, Header(true,false), target column index
dl = DataLoader("datasets/Fish.csv", ',', True, 0)
```


```python
X = dl.getX()
y = dl.gety()
```


```python
X
```




    array([[23.2   , 25.4   , 30.    , 11.52  ,  4.02  ],
           [24.    , 26.3   , 31.2   , 12.48  ,  4.3056],
           [23.9   , 26.5   , 31.1   , 12.3778,  4.6961],
           [26.3   , 29.    , 33.5   , 12.73  ,  4.4555],
           [26.5   , 29.    , 34.    , 12.444 ,  5.134 ],
           [26.8   , 29.7   , 34.7   , 13.6024,  4.9274],
           [26.8   , 29.7   , 34.5   , 14.1795,  5.2785],
           [27.6   , 30.    , 35.    , 12.67  ,  4.69  ],
           [27.6   , 30.    , 35.1   , 14.0049,  4.8438],
           [28.5   , 30.7   , 36.2   , 14.2266,  4.9594],
           [28.4   , 31.    , 36.2   , 14.2628,  5.1042],
           [28.7   , 31.    , 36.2   , 14.3714,  4.8146],
           [29.1   , 31.5   , 36.4   , 13.7592,  4.368 ],
           [29.5   , 32.    , 37.3   , 13.9129,  5.0728],
           [29.4   , 32.    , 37.2   , 14.9544,  5.1708],
           [29.4   , 32.    , 37.2   , 15.438 ,  5.58  ],
           [30.4   , 33.    , 38.3   , 14.8604,  5.2854],
           [30.4   , 33.    , 38.5   , 14.938 ,  5.1975],
           [30.9   , 33.5   , 38.6   , 15.633 ,  5.1338],
           [31.    , 33.5   , 38.7   , 14.4738,  5.7276],
           [31.3   , 34.    , 39.5   , 15.1285,  5.5695],
           [31.4   , 34.    , 39.2   , 15.9936,  5.3704],
           [31.5   , 34.5   , 39.7   , 15.5227,  5.2801],
           [31.8   , 35.    , 40.6   , 15.4686,  6.1306],
           [31.9   , 35.    , 40.5   , 16.2405,  5.589 ],
           [31.8   , 35.    , 40.9   , 16.36  ,  6.0532],
           [32.    , 35.    , 40.6   , 16.3618,  6.09  ],
           [32.7   , 36.    , 41.5   , 16.517 ,  5.8515],
           [32.8   , 36.    , 41.6   , 16.8896,  6.1984],
           [33.5   , 37.    , 42.6   , 18.957 ,  6.603 ],
           [35.    , 38.5   , 44.1   , 18.0369,  6.3063],
           [35.    , 38.5   , 44.    , 18.084 ,  6.292 ],
           [36.2   , 39.5   , 45.3   , 18.7542,  6.7497],
           [37.4   , 41.    , 45.9   , 18.6354,  6.7473],
           [38.    , 41.    , 46.5   , 17.6235,  6.3705],
           [12.9   , 14.1   , 16.2   ,  4.1472,  2.268 ],
           [16.5   , 18.2   , 20.3   ,  5.2983,  2.8217],
           [17.5   , 18.8   , 21.2   ,  5.5756,  2.9044],
           [18.2   , 19.8   , 22.2   ,  5.6166,  3.1746],
           [18.6   , 20.    , 22.2   ,  6.216 ,  3.5742],
           [19.    , 20.5   , 22.8   ,  6.4752,  3.3516],
           [19.1   , 20.8   , 23.1   ,  6.1677,  3.3957],
           [19.4   , 21.    , 23.7   ,  6.1146,  3.2943],
           [20.4   , 22.    , 24.7   ,  5.8045,  3.7544],
           [20.5   , 22.    , 24.3   ,  6.6339,  3.5478],
           [20.5   , 22.5   , 25.3   ,  7.0334,  3.8203],
           [21.    , 22.5   , 25.    ,  6.55  ,  3.325 ],
           [21.1   , 22.5   , 25.    ,  6.4   ,  3.8   ],
           [22.    , 24.    , 27.2   ,  7.5344,  3.8352],
           [22.    , 23.4   , 26.7   ,  6.9153,  3.6312],
           [22.1   , 23.5   , 26.8   ,  7.3968,  4.1272],
           [23.6   , 25.2   , 27.9   ,  7.0866,  3.906 ],
           [24.    , 26.    , 29.2   ,  8.8768,  4.4968],
           [25.    , 27.    , 30.6   ,  8.568 ,  4.7736],
           [29.5   , 31.7   , 35.    ,  9.485 ,  5.355 ],
           [23.6   , 26.    , 28.7   ,  8.3804,  4.2476],
           [24.1   , 26.5   , 29.3   ,  8.1454,  4.2485],
           [25.6   , 28.    , 30.8   ,  8.778 ,  4.6816],
           [28.5   , 31.    , 34.    , 10.744 ,  6.562 ],
           [33.7   , 36.4   , 39.6   , 11.7612,  6.5736],
           [37.3   , 40.    , 43.5   , 12.354 ,  6.525 ],
           [13.5   , 14.7   , 16.5   ,  6.8475,  2.3265],
           [14.3   , 15.5   , 17.4   ,  6.5772,  2.3142],
           [16.3   , 17.7   , 19.8   ,  7.4052,  2.673 ],
           [17.5   , 19.    , 21.3   ,  8.3922,  2.9181],
           [18.4   , 20.    , 22.4   ,  8.8928,  3.2928],
           [19.    , 20.7   , 23.2   ,  8.5376,  3.2944],
           [19.    , 20.7   , 23.2   ,  9.396 ,  3.4104],
           [19.8   , 21.5   , 24.1   ,  9.7364,  3.1571],
           [21.2   , 23.    , 25.8   , 10.3458,  3.6636],
           [23.    , 25.    , 28.    , 11.088 ,  4.144 ],
           [24.    , 26.    , 29.    , 11.368 ,  4.234 ],
           [ 7.5   ,  8.4   ,  8.8   ,  2.112 ,  1.408 ],
           [12.5   , 13.7   , 14.7   ,  3.528 ,  1.9992],
           [13.8   , 15.    , 16.    ,  3.824 ,  2.432 ],
           [15.    , 16.2   , 17.2   ,  4.5924,  2.6316],
           [15.7   , 17.4   , 18.5   ,  4.588 ,  2.9415],
           [16.2   , 18.    , 19.2   ,  5.2224,  3.3216],
           [16.8   , 18.7   , 19.4   ,  5.1992,  3.1234],
           [17.2   , 19.    , 20.2   ,  5.6358,  3.0502],
           [17.8   , 19.6   , 20.8   ,  5.1376,  3.0368],
           [18.2   , 20.    , 21.    ,  5.082 ,  2.772 ],
           [19.    , 21.    , 22.5   ,  5.6925,  3.555 ],
           [19.    , 21.    , 22.5   ,  5.9175,  3.3075],
           [19.    , 21.    , 22.5   ,  5.6925,  3.6675],
           [19.3   , 21.3   , 22.8   ,  6.384 ,  3.534 ],
           [20.    , 22.    , 23.5   ,  6.11  ,  3.4075],
           [20.    , 22.    , 23.5   ,  5.64  ,  3.525 ],
           [20.    , 22.    , 23.5   ,  6.11  ,  3.525 ],
           [20.    , 22.    , 23.5   ,  5.875 ,  3.525 ],
           [20.    , 22.    , 23.5   ,  5.5225,  3.995 ],
           [20.5   , 22.5   , 24.    ,  5.856 ,  3.624 ],
           [20.5   , 22.5   , 24.    ,  6.792 ,  3.624 ],
           [20.7   , 22.7   , 24.2   ,  5.9532,  3.63  ],
           [21.    , 23.    , 24.5   ,  5.2185,  3.626 ],
           [21.5   , 23.5   , 25.    ,  6.275 ,  3.725 ],
           [22.    , 24.    , 25.5   ,  7.293 ,  3.723 ],
           [22.    , 24.    , 25.5   ,  6.375 ,  3.825 ],
           [22.6   , 24.6   , 26.2   ,  6.7334,  4.1658],
           [23.    , 25.    , 26.5   ,  6.4395,  3.6835],
           [23.5   , 25.6   , 27.    ,  6.561 ,  4.239 ],
           [25.    , 26.5   , 28.    ,  7.168 ,  4.144 ],
           [25.2   , 27.3   , 28.7   ,  8.323 ,  5.1373],
           [25.4   , 27.5   , 28.9   ,  7.1672,  4.335 ],
           [25.4   , 27.5   , 28.9   ,  7.0516,  4.335 ],
           [25.4   , 27.5   , 28.9   ,  7.2828,  4.5662],
           [25.9   , 28.    , 29.4   ,  7.8204,  4.2042],
           [26.9   , 28.7   , 30.1   ,  7.5852,  4.6354],
           [27.8   , 30.    , 31.6   ,  7.6156,  4.7716],
           [30.5   , 32.8   , 34.    , 10.03  ,  6.018 ],
           [32.    , 34.5   , 36.5   , 10.2565,  6.3875],
           [32.5   , 35.    , 37.3   , 11.4884,  7.7957],
           [34.    , 36.5   , 39.    , 10.881 ,  6.864 ],
           [34.    , 36.    , 38.3   , 10.6091,  6.7408],
           [34.5   , 37.    , 39.4   , 10.835 ,  6.2646],
           [34.6   , 37.    , 39.3   , 10.5717,  6.3666],
           [36.5   , 39.    , 41.4   , 11.1366,  7.4934],
           [36.5   , 39.    , 41.4   , 11.1366,  6.003 ],
           [36.6   , 39.    , 41.3   , 12.4313,  7.3514],
           [36.9   , 40.    , 42.3   , 11.9286,  7.1064],
           [37.    , 40.    , 42.5   , 11.73  ,  7.225 ],
           [37.    , 40.    , 42.4   , 12.3808,  7.4624],
           [37.1   , 40.    , 42.5   , 11.135 ,  6.63  ],
           [39.    , 42.    , 44.6   , 12.8002,  6.8684],
           [39.8   , 43.    , 45.2   , 11.9328,  7.2772],
           [40.1   , 43.    , 45.5   , 12.5125,  7.4165],
           [40.2   , 43.5   , 46.    , 12.604 ,  8.142 ],
           [41.1   , 44.    , 46.6   , 12.4888,  7.5958],
           [30.    , 32.3   , 34.8   ,  5.568 ,  3.3756],
           [31.7   , 34.    , 37.8   ,  5.7078,  4.158 ],
           [32.7   , 35.    , 38.8   ,  5.9364,  4.3844],
           [34.8   , 37.3   , 39.8   ,  6.2884,  4.0198],
           [35.5   , 38.    , 40.5   ,  7.29  ,  4.5765],
           [36.    , 38.5   , 41.    ,  6.396 ,  3.977 ],
           [40.    , 42.5   , 45.5   ,  7.28  ,  4.3225],
           [40.    , 42.5   , 45.5   ,  6.825 ,  4.459 ],
           [40.1   , 43.    , 45.8   ,  7.786 ,  5.1296],
           [42.    , 45.    , 48.    ,  6.96  ,  4.896 ],
           [43.2   , 46.    , 48.7   ,  7.792 ,  4.87  ],
           [44.8   , 48.    , 51.2   ,  7.68  ,  5.376 ],
           [48.3   , 51.7   , 55.1   ,  8.9262,  6.1712],
           [52.    , 56.    , 59.7   , 10.6863,  6.9849],
           [56.    , 60.    , 64.    ,  9.6   ,  6.144 ],
           [56.    , 60.    , 64.    ,  9.6   ,  6.144 ],
           [59.    , 63.4   , 68.    , 10.812 ,  7.48  ],
           [ 9.3   ,  9.8   , 10.8   ,  1.7388,  1.0476],
           [10.    , 10.5   , 11.6   ,  1.972 ,  1.16  ],
           [10.1   , 10.6   , 11.6   ,  1.7284,  1.1484],
           [10.4   , 11.    , 12.    ,  2.196 ,  1.38  ],
           [10.7   , 11.2   , 12.4   ,  2.0832,  1.2772],
           [10.8   , 11.3   , 12.6   ,  1.9782,  1.2852],
           [11.3   , 11.8   , 13.1   ,  2.2139,  1.2838],
           [11.3   , 11.8   , 13.1   ,  2.2139,  1.1659],
           [11.4   , 12.    , 13.2   ,  2.2044,  1.1484],
           [11.5   , 12.2   , 13.4   ,  2.0904,  1.3936],
           [11.7   , 12.4   , 13.5   ,  2.43  ,  1.269 ],
           [12.1   , 13.    , 13.8   ,  2.277 ,  1.2558],
           [13.2   , 14.3   , 15.2   ,  2.8728,  2.0672],
           [13.8   , 15.    , 16.2   ,  2.9322,  1.8792]])




```python
y
```




    array([ 242. ,  290. ,  340. ,  363. ,  430. ,  450. ,  500. ,  390. ,
            450. ,  500. ,  475. ,  500. ,  500. ,  340. ,  600. ,  600. ,
            700. ,  700. ,  610. ,  650. ,  575. ,  685. ,  620. ,  680. ,
            700. ,  725. ,  720. ,  714. ,  850. , 1000. ,  920. ,  955. ,
            925. ,  975. ,  950. ,   40. ,   69. ,   78. ,   87. ,  120. ,
            100. ,  110. ,  120. ,  150. ,  145. ,  160. ,  140. ,  160. ,
            169. ,  161. ,  200. ,  180. ,  290. ,  272. ,  390. ,  270. ,
            270. ,  306. ,  540. ,  800. , 1000. ,   55. ,   60. ,   90. ,
            120. ,  150. ,  140. ,  170. ,  145. ,  200. ,  273. ,  300. ,
              5.9,   32. ,   40. ,   51.5,   70. ,  100. ,   78. ,   80. ,
             85. ,   85. ,  110. ,  115. ,  125. ,  130. ,  120. ,  120. ,
            130. ,  135. ,  110. ,  130. ,  150. ,  145. ,  150. ,  170. ,
            225. ,  145. ,  188. ,  180. ,  197. ,  218. ,  300. ,  260. ,
            265. ,  250. ,  250. ,  300. ,  320. ,  514. ,  556. ,  840. ,
            685. ,  700. ,  700. ,  690. ,  900. ,  650. ,  820. ,  850. ,
            900. , 1015. ,  820. , 1100. , 1000. , 1100. , 1000. , 1000. ,
            200. ,  300. ,  300. ,  300. ,  430. ,  345. ,  456. ,  510. ,
            540. ,  500. ,  567. ,  770. ,  950. , 1250. , 1600. , 1550. ,
           1650. ,    6.7,    7.5,    7. ,    9.7,    9.8,    8.7,   10. ,
              9.9,    9.8,   12.2,   13.4,   12.2,   19.7,   19.9])



# Linear Regression


```python
# Add Bias term before training
X = DataLoader.add_constant(X)

# Split data into train and test
X_train, y_train, X_test, y_test = DataLoader.trainTestSplit(X,y,0.2)

# Create regression object
linreg = LinearRegression()

# Train the model
linreg.fit(X_train, y_train)

# Output results
print(f'R2 Score on test set: {linreg.score(X_test, y_test)}\n')
print(f'Coefficients for linear model: {linreg.coefficients}')
```

    R2 Score on test set: 0.8700421130641914
    
    Coefficients for linear model: [-516.76987327   40.57633659    3.98457838  -22.3904408    23.66399307
       48.62068414]


# Ridge Regression


```python
X = dl.getX()
X = DataLoader.add_constant(X)
y = dl.gety()

# Train test split
X_train, y_train, X_test, y_test = DataLoader.trainTestSplit(X,y,0.2)

ridge = RidgeRegression()

# Params: X, y, alpha (penalty coefficient)
ridge.fit(X_train, y_train, 20)

# Output Results
print(f'R2 Score on test set: {ridge.score(X_test, y_test)}\n')
print(f'Coefficients for linear model: {ridge.coefficients}')
```

    R2 Score on test set: 0.8121391150132838
    
    Coefficients for linear model: [-214.04173121   53.92347905   10.16001537  -40.09192144   30.6868183
      -17.40733713]


# Logistic Regression


```python
# Load in data
dl = DataLoader("datasets/voice.csv", ',',True, 20)
X = dl.getX()
y = dl.gety()

# Add bias term to features
X = DataLoader.add_constant(X)

# Create regression object
logistic = LogisticRegression()

# Params: X, y, start_theta, learning rate, iterations
logistic.fit(X, y, np.zeros(X.shape[1]), 0.02, 1000)

print(f'Coefficients for linear model: {logistic.coefficients}')
```

    Coefficients for linear model: [ 4.56118030e-01  1.40754862e-02  8.74969266e-02  1.46770888e-02
     -1.33709863e-01  1.32249688e-01  2.65959551e-01 -1.35551320e+00
      2.41488704e-01  5.98845795e-01  6.47911842e-01  2.55632494e-02
      1.40754862e-02 -1.83311083e-01  3.12970580e-03  8.10604034e-02
     -1.33784484e-02 -8.34245285e-02 -8.43412273e-02 -9.16698852e-04
      7.75575195e-02]



```python
# Female Test case, grabbed from data
test_female = np.array([0.196623438573098,0.0396833940839021,0.199588759424263,0.188649760109664,0.213598355037697,0.0249485949280329,3.21840072313148,14.2664514713572,0.8427271293845,0.264958086407542,0.200164496230295,0.196623438573098,0.187679540165102,0.0636942675159236,0.256410256410256,0.5224609375,0.1904296875,3.6474609375,3.45703125,0.163912429378531]).reshape(1,-1)
# Add bias term
test_female = DataLoader.add_constant(test_female)
# Reshape because function expects a column vector
test_female=test_female.reshape(-1,1)

# Make prediction
logistic.predict(test_female)
```




    0




```python
# Male Test case, grabbed from data
test_male = np.array([0.190846297652897,0.0657902823878029,0.207950987066031,0.132280462899932,0.244356705241661,0.112076242341729,1.56230368302863,7.83434988641618,0.938546013440116,0.538809584118407,0.0501293396868618,0.190846297652897,0.113322801479182,0.0175438596491228,0.275862068965517,1.43411458333333,0.0078125,6.3203125,6.3125,0.25477978832366]).reshape(1,-1)
# Add bias term
test_male = DataLoader.add_constant(test_male)
# Reshape because function expects a column vector
test_male=test_male.reshape(-1,1)

# Make prediction
logistic.predict(test_male)
```




    1



# General Documentation


```python
help(DaveML)
```

    Help on module DaveML:
    
    NAME
        DaveML - Welcome to my simple ML library implemented in C++
    
    CLASSES
        pybind11_builtins.pybind11_object(builtins.object)
            DataLoader
            LinearRegression
                RidgeRegression
            LogisticRegression
        
        class DataLoader(pybind11_builtins.pybind11_object)
         |  Method resolution order:
         |      DataLoader
         |      pybind11_builtins.pybind11_object
         |      builtins.object
         |  
         |  Methods defined here:
         |  
         |  __init__(...)
         |      __init__(self: DaveML.DataLoader, arg0: str, arg1: str, arg2: bool, arg3: int) -> None
         |  
         |  add_constant(...)
         |      add_constant(self: numpy.ndarray[numpy.float64[m, n]]) -> numpy.ndarray[numpy.float64[m, n]]
         |  
         |  getX(...)
         |      getX(self: DaveML.DataLoader) -> numpy.ndarray[numpy.float64[m, n]]
         |  
         |  gety(...)
         |      gety(self: DaveML.DataLoader) -> numpy.ndarray[numpy.float64[m, 1]]
         |  
         |  standardizeFeatures(...)
         |      standardizeFeatures(self: numpy.ndarray[numpy.float64[m, n]]) -> numpy.ndarray[numpy.float64[m, n]]
         |  
         |  trainTestSplit(...)
         |      trainTestSplit(self: numpy.ndarray[numpy.float64[m, n]], arg0: numpy.ndarray[numpy.float64[m, 1]], arg1: float) -> Tuple[numpy.ndarray[numpy.float64[m, n]], numpy.ndarray[numpy.float64[m, 1]], numpy.ndarray[numpy.float64[m, n]], numpy.ndarray[numpy.float64[m, 1]]]
         |  
         |  ----------------------------------------------------------------------
         |  Static methods inherited from pybind11_builtins.pybind11_object:
         |  
         |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
         |      Create and return a new object.  See help(type) for accurate signature.
        
        class LinearRegression(pybind11_builtins.pybind11_object)
         |  Method resolution order:
         |      LinearRegression
         |      pybind11_builtins.pybind11_object
         |      builtins.object
         |  
         |  Methods defined here:
         |  
         |  __init__(...)
         |      __init__(self: DaveML.LinearRegression) -> None
         |  
         |  fit(...)
         |      fit(self: DaveML.LinearRegression, arg0: numpy.ndarray[numpy.float64[m, n]], arg1: numpy.ndarray[numpy.float64[m, 1]]) -> None
         |  
         |  predict(...)
         |      predict(self: DaveML.LinearRegression, arg0: numpy.ndarray[numpy.float64[m, n]]) -> numpy.ndarray[numpy.float64[m, 1]]
         |  
         |  score(...)
         |      score(self: DaveML.LinearRegression, arg0: numpy.ndarray[numpy.float64[m, n]], arg1: numpy.ndarray[numpy.float64[m, 1]]) -> float
         |  
         |  ----------------------------------------------------------------------
         |  Readonly properties defined here:
         |  
         |  coefficients
         |  
         |  ----------------------------------------------------------------------
         |  Static methods inherited from pybind11_builtins.pybind11_object:
         |  
         |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
         |      Create and return a new object.  See help(type) for accurate signature.
        
        class LogisticRegression(pybind11_builtins.pybind11_object)
         |  Method resolution order:
         |      LogisticRegression
         |      pybind11_builtins.pybind11_object
         |      builtins.object
         |  
         |  Methods defined here:
         |  
         |  __init__(...)
         |      __init__(self: DaveML.LogisticRegression) -> None
         |  
         |  fit(...)
         |      fit(self: DaveML.LogisticRegression, arg0: numpy.ndarray[numpy.float64[m, n]], arg1: numpy.ndarray[numpy.float64[m, 1]], arg2: numpy.ndarray[numpy.float64[m, 1]], arg3: float, arg4: int) -> None
         |  
         |  gradient_cost(...)
         |      gradient_cost(self: numpy.ndarray[numpy.float64[m, 1]], arg0: numpy.ndarray[numpy.float64[m, n]], arg1: numpy.ndarray[numpy.float64[m, 1]]) -> numpy.ndarray[numpy.float64[m, 1]]
         |  
         |  predict(...)
         |      predict(self: DaveML.LogisticRegression, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> int
         |  
         |  sigmoid(...)
         |      sigmoid(self: numpy.ndarray[numpy.float64[m, n]]) -> numpy.ndarray[numpy.float64[m, n]]
         |  
         |  ----------------------------------------------------------------------
         |  Readonly properties defined here:
         |  
         |  coefficients
         |  
         |  ----------------------------------------------------------------------
         |  Static methods inherited from pybind11_builtins.pybind11_object:
         |  
         |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
         |      Create and return a new object.  See help(type) for accurate signature.
        
        class RidgeRegression(LinearRegression)
         |  Method resolution order:
         |      RidgeRegression
         |      LinearRegression
         |      pybind11_builtins.pybind11_object
         |      builtins.object
         |  
         |  Methods defined here:
         |  
         |  __init__(...)
         |      __init__(self: DaveML.RidgeRegression) -> None
         |  
         |  fit(...)
         |      fit(self: DaveML.RidgeRegression, arg0: numpy.ndarray[numpy.float64[m, n]], arg1: numpy.ndarray[numpy.float64[m, 1]], arg2: float) -> None
         |  
         |  ----------------------------------------------------------------------
         |  Methods inherited from LinearRegression:
         |  
         |  predict(...)
         |      predict(self: DaveML.LinearRegression, arg0: numpy.ndarray[numpy.float64[m, n]]) -> numpy.ndarray[numpy.float64[m, 1]]
         |  
         |  score(...)
         |      score(self: DaveML.LinearRegression, arg0: numpy.ndarray[numpy.float64[m, n]], arg1: numpy.ndarray[numpy.float64[m, 1]]) -> float
         |  
         |  ----------------------------------------------------------------------
         |  Readonly properties inherited from LinearRegression:
         |  
         |  coefficients
         |  
         |  ----------------------------------------------------------------------
         |  Static methods inherited from pybind11_builtins.pybind11_object:
         |  
         |  __new__(*args, **kwargs) from pybind11_builtins.pybind11_type
         |      Create and return a new object.  See help(type) for accurate signature.
    
    FILE
        /Users/dpogrebitskiy/Documents/Programming/Fall 2022/CS3520/monorepo-pogrebitskiy/Assignment12_Project/part1/DaveML.so
    
    

