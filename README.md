# ADformer: Adaptive theory combined with Dictionary-guided Transformer model for multivariate time Series Anomaly Detection


## 1. Get Started--Environment configuration

 Install >= Python 3.6, PyTorch >= 1.4.0. cuda vision = 12.8. pandas >= 1.5.0.pytorvh cuda = 12.1。 scikit learn >= 1.6.1

## 2. Datasets used
  Download data. You can obtain the benchmark datasets from the Github Repository of DCdetector ( [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR) ).

### 2.1 Datasets information:

![Data Information](AD-Model/img/data-information.png)


## 3. Model structure

### 3.1 Overview Architecture
![Model structure](AD-Model/img/Model_Architecture.png)

### 3.2 Adaptive Dynamic Neighbor Mask (ADMN)
![Model structure](AD-Model/img/ADNM_Masking_handmake.png  )

### 3.3 Dictionaries-based Cross-attention
![Model structure](AD-Model/img/Dictionary-based-right.png)

## 4. Experiment results reproduce

**1.** Train and Test. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:

```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/SWaT.sh
bash ./scripts/PSM.sh
```


## 5. Experiment results

### 5.1 Implementation Details



### 5.2 Evaluation Metrics

### 1) Precision

**Precision** measures the proportion of correctly predicted anomalies among all predicted anomalies:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Where:
- **TP** = True Positives (correctly detected anomalies)
- **FP** = False Positives (normal points mistakenly classified as anomalies)

---

### 2) Recall

**Recall** measures the proportion of actual anomalies that are correctly identified:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Where:
- **FN** = False Negatives (anomalies the model failed to detect)

---

### 3) F1-score

**F1-score** is the harmonic mean of Precision and Recall:

$$
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$


### 5.3 Baseline model

Based on the code provided in the paper [GD Former](https://arxiv.org/abs/2501.18196), we replicated the model on our own local device as our baseline model, with the results shown in the following table.

![Baseline result](AD-Model/img/difference.png)

It can be seen that the baseline model results are almost identical to those in the paper and that the algorithm performs best on the PSM dataset in aggregate.

### 5.4 Main result

![compare_with_16_methods-table](AD-Model/img/compare_with_16_methods-table.png)

### 5.5.1 Parameter Sensitivity

### 1) ADMN-hyperparameter-pearson_coefficient:
![Parameter Sensitivity_pearson](AD-Model/img/pearson_coefficient_f1.png)


### 2) Parameter Sensitivity 1-4:

![Parameter Sensitivity14](AD-Model/img/param_sensitivity_for1-4.png)

### 3) Parameter Sensitivity 5-9:

![Parameter Sensitivity59](AD-Model/img/param_sensitivity_for5-9.png)

### 5.5.2 Ablation Experiments

## Ablation Study: Module Replacements

### a. Adaptive Dynamic Neighbor Masking (ADNM)
**Replacement:** Replaced ADNM with the traditional causal mask (upper-triangular), as used in standard Transformers. This restricts attention to only past timesteps and lacks adaptive masking based on input content.
![ADNM Mask](AD-Model/img/Trangle%20Mask%20vs.%20ADNM%20Mask.png)

---

### b. Dictionary-based Cross Attention
**Replacement:** 
Replaced the dictionary-based cross-attention with conventional self-attention, where queries, keys, and values are derived from the same input sequence. This eliminates the use of global shared prototypes.
![Cross Attention](AD-Model/img/Traditional-Trasformer%20Self-Attention%20vs.%20Dictionary-based%20cross-attention.png)


---

### c. Dynamic Thresholding via SPOT
**Replacement:** Replaced SPOT (based on Extreme Value Theory) with a static percentile thresholding strategy (e.g., Top-5%), where anomaly scores are truncated at a fixed quantile level.

![SPOT Thresholding](AD-Model/img/SPOT.png)

### 5.5.3 Model efficiency

![Model Efficiency](AD-Model/img/Model%20Efficiency%20Comparison.png)


## 6. Reference

> DCdetector: https://github.com/DAMO-DI-ML/KDD2023-DCdetector
> 
> GDformer: https://github.com/yuppielqx/GDformer
> 
> OmniAnomaly： https://github.com/NetManAIOps/OmniAnomaly
> 
> TranAD: https://github.com/imperial-qore/TranAD
> 
> DDMT: https://github.com/Lelantos42/DDMT

## 7. Member Contribution:

Heqing Shi(ID: Stardust752121): Project Management, Method Improvement and Theory Verification, Code Implementation and Debugging, Collection of Experimental Results.

Yansong Shi (ID: Patrick-Shi): Data acquisition, data preprocessing, baseline model evaluation ( Merge Commit ID 0335bad) and building experimental charts/tables .

Xiaolong Ge (ID: EmiyaKatuz): project environment construction, code implementation and debugging, experimental results collection, and experimental results visualisation.

Geyang Zhang: project environment construction, code implementation and debugging, experimental results collection, and experimental results visualisation.

Guanjin Wang: Method improvement and theory verification, slide production.
