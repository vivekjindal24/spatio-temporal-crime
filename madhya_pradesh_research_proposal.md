# An Intelligent Computational Framework Using Spatiotemporal Prediction for Crimes Against Women in the Madhya Pradesh Region

Shruti Sharma1 and Manoj Rawat1  
1Department of Computer Science and Engineering, Medicaps University, Indore, Madhya Pradesh, India  
Email: {en24cs6e10016, manoj.rawat}@medicaps.ac.in

## Abstract

Forecasting crimes against women is a socially important and technically challenging problem. Current crime prediction models work well for some cities with detailed location data but perform poorly when data is aggregated or imprecise. This research aims to design a deep learning model that can learn patterns from spatiotemporal crime data, focusing on crime records from the Madhya Pradesh region that are often provided at district level. The model will be compared against established benchmarks and evaluated for accuracy. The final objective is to provide a predictive tool that can help law enforcement and policymakers allocate resources more effectively to prevent women-centric crimes.

## 1 Introduction and Background

Crime prediction refers to the use of historical crime data to forecast future crime occurrences in different areas at specific times. With increasing digitization of crime records, there is potential to use machine learning and deep learning to extract patterns that human analysts might miss.

However, most well-known crime prediction models assume detailed spatial information, such as exact latitude and longitude for each crime incident. In India, official crime data from the National Crime Records Bureau (NCRB) is usually available at broader geographic levels such as districts or cities. For example, records will show the number of reported rape, assault, or kidnapping cases in a district in a given year but not the precise coordinates of each incident.

The main challenge is to design a model that can work effectively with such aggregated data while still capturing meaningful spatial and temporal patterns. The work will build on existing research in crime forecasting, with a focus on models that combine spatial relationships with temporal patterns.

## 2 Literature Review

[1] paper proposes a hybrid model that integrates two components:

1. **Informer**: an attention-based model that is designed to learn temporal patterns over long periods, such as weeks or months of crime counts.
2. **Spatio Temporal Graph Convolutional Network (ST-GCN)**: a method that treats different regions (such as neighborhoods or districts) as nodes in a graph and learns how crime in one region relates to crime in neighboring regions.

In this work, this model was applied to four crime types in Chicago, including assault and theft. The model predicted future crime counts based on past data, and it performed better than older approaches like simple recurrent neural networks. While this research shows strong results, it relies on high-resolution data with exact locations and multiple years of incident records. This level of detail is not available for many Indian datasets.

A group of researchers presented a hybrid Informer and Spatio Temporal Graph Convolutional Network (ST-GCN) that integrates graph structures with powerful temporal sequence learning to forecast crime types in Chicago, underscoring the effectiveness of hybrid deep architectures for spatiotemporal data fusion [1]. [2] demonstrates that incorporating spatiotemporal lag variables into machine learning frameworks significantly improves crime risk prediction by capturing autocorrelation in crime distributions.

[3] introduced MRAGNN, a multi-type relations-aware graph neural network that combines spatial and type-temporal dependencies for crime occurrence prediction and addresses imbalance via focal loss, leading to superior classification performance. While [4] proposed Ada-GCNLSTM, enhancing spatial feature extraction via graph convolutional structures combined with LSTM to model temporal dependencies, achieving consistent improvements across multiple urban datasets.

Beyond pure GNNs, attention-based models such as ACSAformer blend sparse attention with adaptive graph convolution to capture complex inter-feature dependencies in long sequences, which is critical for fine-grained crime forecasting [5]. At street level, [6] demonstrated that Graph Attention Networks (GAT) capture street network dependencies effectively, outperforming standard GCN approaches for incident prediction.

In addition to deep models, hybrid models combining CNN and LSTM have shown strong performance for spatiotemporal pattern extraction in crime counts [7]. Studies exploring auxiliary features like park event density show that integrating human activity indicators can further enhance hotspot prediction accuracy beyond traditional spatiotemporal methods [8].

Several surveys highlight the evolution and effectiveness of spatiotemporal graph neural networks (STGNNs) in predictive tasks [9]. Research on crime forecasting in non-Western contexts, such as Costa Rica, underscores that spatial correlations between regions significantly enhance model performance when predicting regional crime counts [10].

However, many of these methods have been tested on data with precise spatial coordinates or mobility information that may not be present in regional Indian crime data. This gap motivates the need for a model that can work with coarser spatial data, such as district-level crime counts. The last decade has seen significant progress in crime forecasting models that combine spatial and temporal learning.

## 3 Problem Statement

Crime data for the Madhya Pradesh region is primarily available as aggregated numbers by district, city, and year. Existing spatiotemporal models perform well on precise incident datasets but are not designed for data aggregated over regions. Predicting women-centric crimes such as rape, assault, and domestic violence using such data requires adapting and extending existing deep learning frameworks to work with the available regional datasets.

## 4 Research Objectives

The objectives of this research are:

1. To review and analyze existing deep learning models used for crime prediction, with a focus on hybrid spatiotemporal methods.
2. To organize and preprocess crime data for the Madhya Pradesh region, including NCRB-aligned records, so that it can be used for machine learning models.
3. To adapt the hybrid architecture so that it can work with aggregated spatial data rather than exact incident locations and to incorporate district adjacency and socioeconomic features into a spatiotemporal model.
4. To evaluate the proposed model’s performance using standard metrics and compare it with baseline models.
5. To benchmark the model using the Chicago crime dataset to validate its effectiveness on fine-grained spatial data.
6. To produce a reliable predictive tool that can forecast women-centric crime trends in the Madhya Pradesh region.

## 5 Proposed Research

The proposed methodology builds on the below-mentioned core ideas while making changes to suit the crime reporting format of the Madhya Pradesh region.

This research will use a modified version of the hybrid model [1] that can work on aggregated data (such as district yearly crime counts). The proposed method will:

- Use district or city crime counts from NCRB and regional records as time series input.
- Construct an adjacency graph representing spatial relationships between districts (e.g., neighboring areas, shared boundaries, or socioeconomic similarity).
- Use a transformer or attention-based temporal model (like Informer) to learn trends over time.
- Use a graph neural network to capture spatial relationships.
- Combine the temporal and spatial outputs to produce future predictions for women-centric crime categories.
- Optimize model parameters using a computational intelligence optimizer (such as particle swarm optimization) to improve performance.

The key difference from the base paper is that this model will work on aggregated data and will be tailored for the crime reporting format of the Madhya Pradesh region.

## 6 Datasets

The main dataset will be National Crime Records Bureau (NCRB) data and related regional crime records for women-centric crimes in the Madhya Pradesh region. This includes reported counts of rape, assault, and other relevant categories by district and year.

For benchmarking and comparison, the Chicago crime dataset containing exact incident records with latitude and longitude will be used. This allows comparison with methods that work on precise spatial data.

## 7 Timeline

| Phase | Duration |
|---|---|
| Literature review and dataset collection | 2 Months |
| Data preprocessing | 2 Months |
| Model design and implementation | 3 Months |
| Experiments and optimization | 3.5 Months |
| Evaluation and analysis | 2 Months |
| Writing and final submission | 4.5 Months |

## 8 Conclusion

This research aims to build a deep learning model that can forecast women-centric crimes using aggregated crime data from the Madhya Pradesh region. By adapting hybrid spatiotemporal architectures and optimizing them for aggregated inputs, the work will fill a gap in current crime forecasting applications. The final result will be a predictive tool suitable for policy and enforcement planning.

## References

[1] Y. Fan, X. Hu, and J. Hu, “Research on a Crime Spatiotemporal Prediction Method Integrating Informer and ST-GCN: A Case Study of Four Crime Types in Chicago,” *Big Data and Cognitive Computing*, vol. 9, no. 7, 2025.

[2] Y. Deng, R. He, and Y. Liu, “Crime risk prediction incorporating geographical spatiotemporal dependency into machine learning models,” *Information Sciences*, vol. 646, p. 119414, 2023. Available: <https://www.sciencedirect.com/science/article/pii/S0020025523009994>

[3] S. Wang, Y. Zhang, X. Piao, X. Lin, Y. Hu, and B. Yin, “MRAGNN: Refining urban spatio-temporal prediction of crime occurrence with multi-type crime correlation learning,” *Expert Systems with Applications*, vol. 265, 2025.

[4] M. Shan, C. Ye, P. Chen, and S. Peng, “Ada-GCNLSTM: An adaptive urban crime spatiotemporal prediction model,” *Journal of Safety Science and Resilience*, vol. 6, no. 2, pp. 226–236, 2025.

[5] Z. Qin, B. Wei, C. Gao, F. Zhu, W. Qin, and Q. Zhang, “ACSAformer: A crime forecasting model based on sparse attention and adaptive graph convolution,” *Frontiers in Physics*, vol. 13, 2025.

[6] J. Sui, P. Chen, and H. Gu, “Deep Spatio-Temporal Graph Attention Network for Street-Level 110 Call Incident Prediction,” *Applied Sciences (Switzerland)*, vol. 14, no. 20, 2024.

[7] L. Mao, W. Du, S. Wen, Q. Li, T. Zhang, and W. Zhong, “Crime forecasting: A spatio-temporal analysis with deep learning models,” *Journal of Computational Methods in Sciences and Engineering*, vol. 25, no. 5, pp. 4090–4099, 2025. Available: <https://doi.org/10.1177/14727978251337993>

[8] T. C. Hakyemez and B. Badur, “Incorporating park events into crime hotspot prediction on street networks: A spatiotemporal graph learning approach,” *Applied Soft Computing*, vol. 148, p. 110886, 2023.

[9] Y. Wang, “Advances in spatiotemporal graph neural network prediction research,” *International Journal of Digital Earth*, vol. 16, no. 1, pp. 2034–2066, 2023.

[10] M. Solís and L. A. Calvo-Valverde, “Deep Learning for Crime Forecasting of Multiple Regions, Considering Spatial–Temporal Correlations between Regions,” *Engineering Proceedings*, vol. 68, no. 1, 2024.
