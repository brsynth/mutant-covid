README

This Git repository contains all codes used for classification (prognosis and diagnosis) as well as Generalized Additive Mixed Models (GAMMs) analyses conducted in Ahavi et al. [link biorxiv].

Regarding classification models (Python codes): 

(i) "growth_parameters_main" is the code used to perform growth-parameter-based classification. Using on a growth parameter table, it trains four models for each strain in order to perform COVID-19 prognosis and diagnosis: a support vector machine (SVM), a logistic regression model, a gradient-boosted tree model XGBoost, and a soft-voting ensemble combining the three base classifiers.

(ii) "time_series_model" and time_series_main are the codes used to perform time-series classification. They train a one-dimensional CNN (CNN1D) and a TCN on the raw growth curve for each strain. This code also allows the user to add the first and second derivatives of the raw growth curve as inputs for the model to better capture growth dynamics. Channel-wise z-score normalization can be applied to raw curves and/or derivatives separately.

(iii) "FiLM.model" and "FiLM.main" are the codes used for context-aware classification using Feature-wise Linear Modulation (FiLM). For each strain, FiLM parameters were acquired through a multilayer perceptron fed with the normalized growth parameters. These parameters were then used to modulate the feature maps of the corresponding growth curve through an affine transformation. Two models are trained: a FiLM-CNN1D and a FiLM-TCN. In the FiLM-CNN1D, FiLM is applied after concatenation of the two parallel one-dimensional convolution branches, whereas in the FiLM-TCN, FiLM is applied after the second temporal convolution block.

(iv) "early_fusion_model" and "early_fusion_main" are the codes used for early-fusion multistrain classification. They take as input the input configuration maximizing the balanced accuracy for each strain tested. Each input is a new channel and FiLM using growth parameters can be applied only to the corresponding time series. They then train a CNN1D and a TCN on the combined list of inputs.

(v) "late_fusion_model" and "late_fusion_main" are the codes used for late-fusion multistrain classification. For each strain included in the final model, they train separately the submodel maximizing the balanced accuracy. Then they train a weighted soft-voting ensemble model to perform classification. Fusion weights were learned independently within each outer fold in order to prevent data leakage.

Additional design details are available in Ahavi et al. [link biorxiv]. Briefly, a nested patient-level cross-validation strategy with 5 outer folds and 3 inner folds is implemented to prevent data leakage. The same stratified splits are used across all models to ensure comparability. Hyperparameter tuning had balanced accuracy maximization as objective. Optuna was used for time-series hyperparameter search.

Concerning GAMMs (R codes):

(i) GAMM_all_mutants_model_comparison is the code used to select the best model for each classification problem and each screened strain among three candidate models of increasing complexity. The selected model is the one with the lowest Akaike Information Criterion (AIC). For models using basis dimensions (k), each basis dimension is optimized sequentially using the k.check() function. The basis dimension selected was the lowest one from 3 to 20 yielding a k.check() superior to 0.9. If no basis dimension fulfills this condition, then its value was set to 20.

(ii) GAMM_selected_mutants_model_comparison is the code used to select the best model for each classification problem and each strain tested during the validation phase. It relies on the same logic as the GAMM_all_mutants_model_comparison code.

(iii) GAMM_all_mutants is the code used to perform the statistical test during mutant screening for each classification problem, based on the model selected by the GAMM_all_mutants_model_comparison code. When two classes are considered statistically different, it computes the time window during which the difference occurs. It also computes the simultaneous 95% confidence bands and the pointwise standard deviation of model-predicted values across replicate/curve units within each group.

(iv) GAMM_selected_mutants is the code used to perform the statistical test during mutant validation for each classification problem, based on the model selected by the GAMM_selected_mutants_model_comparison code. It follows the same logic as the GAMM_all_mutants code.

Additional design details are available in Ahavi et al. [link biorxiv]. The bam() function from the mgcv package (v.1.9.3) was used to fit the growth curves.