# Machine Learning Models to predict daily actual evapotranspiration of citrus orchards
## A. Pagano; D. Croce; F. Amato; M. Ippolito; D. De Caro; A. Motisi; G. Provenzano; I. Tinnirello

Precise estimation of actual evapotranspiration is essential for various hydrological processes, including those related to agricultural water source management. 
Indeed, the increasing demands of agricultural production, coupled with increasingly frequent drought events in many parts of the world, necessitate a more careful assessment of irrigation needs. 

Artificial Intelligence-based models represent a promising alternative to the most common estimation techniques. 
In this context, the main challenges are choosing the best possible model and selecting the most representative features. 
The objective of this research was to evaluate two different machine learning algorithms, namely Multi-Layer Perceptron (MLP) and Random Forest (RF), to predict daily actual evapotranspiration in a Mediterranean citrus orchard using different feature combinations. With many features available coming from the various infield sensors, a thorough analysis was performed to measure feature importance (FI), scatter matrix observations, and Pearson's correlation coefficient calculation, which resulted in the selection of 12 promising feature combinations. 
Overall, 24 different models were developed and compared, evaluating the performance of the prediction algorithm (both for RF or MLP) and the importance of the different input variables adopted. 

Results show that the accuracy of the proposed machine learning models remains acceptable even when the number of input features is reduced from 10 to 4. 
Among the different ML algorithms developed, the best performance was achieved by the Random Forest method when using seven input features. 
In this case, the values of the root mean square error (RMSE) and coefficient of determination (R2) associated with cross-validation were 0.39 mm/d and 0.84, respectively. 

Finally, the results obtained show that the joint use of agro-meteorological and remote sensing data improves the performance of evapotranspiration forecasts compared with models using only weather variables. 

### Structure
Be sure to have this structure in your project:
ROOT
|— PAPER
|  |—DRAFTS
|  |—BIBLIOGRAPHY
|  
|— eta_ml
|   |- data
|   |   |- external
|   |   |- raw
|   |   |- interim
|   |   |- processed
|   |   |- predicted
|   |
|   |- models
|   |
|   |- prediction
|   |
|   |- visualization
|   |
|   |- eta_ml
|   |   |- data
|   |   |- models
|   |   |- prediction
|   |   |- visualization
|   |
