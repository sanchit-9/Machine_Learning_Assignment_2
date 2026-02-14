# Machine Learning - Assignment 2 
### Submitted By: Sanchit Bathla (2025AA05922)
### (2025aa05922@wilp.bits-pilani.ac.in)

a. **Problem statement**

    The problem is to understand, analyze, and predict viral content across major social media platforms (TikTok, Instagram, X (Twitter), YouTube Shorts) using engagement metrics, sentiment signals, and content attributes. The goal is to develop machine learning models that can classify content as viral or not viral based on these features.

b. **Dataset description** [1 mark]

    This dataset is designed to help data scientists, analysts, and researchers understand, analyze, and predict viral content across major social media platforms. It captures realistic engagement behavior, sentiment signals, and content attributes that influence virality in today’s digital ecosystem. It includes multi-platform data from TikTok, Instagram, X (Twitter), and YouTube Shorts, with consistent metrics for cross-platform comparison. Key features include post metadata (post_id, platform, content_type, topic, language, region), time and trend signals (post_datetime), hashtags and sentiment (hashtags, sentiment_score), engagement metrics (views, likes, comments, shares), and engineered features (engagement_rate, is_viral).

c. **Models used**:

    Make a Comparison Table with the evaluation metrics calculated for all the 6 models as below:

    | ML Model Name            | Accuracy | AUC                  | Precision          | Recall             | F1                 | MCC                |
    | :--------------------    | :------- | :------------------- | :----------------- | :----------------- | :----------------- | :----------------- |
    | Logistic Regression      | 0.98     | 0.9985618267735025   | 0.9926739926739927 | 0.9783393501805054 | 0.9854545454545455 | 0.9537182122405866 |
    | Decision Tree            | 1        | 1                    | 1                  | 1                  | 1                  | 1                  |
    | kNN                      | 0.8775   | 0.9294561357165918   | 0.8986013986013986 | 0.927797833935018  | 0.9129662522202486 | 0.707422845622539  |
    | Naive Bayes              | 0.9425   | 0.9983563734554314   | 0.9233333333333333 | 1                  | 0.9601386481802426 | 0.8664164742057181 |
    | Random Forest (Ensemble) | 1        | 1                    | 1                  | 1                  | 1                  | 1                  |
    | XGBoost (Ensemble)       | 1        | 1                    | 1                  | 1                  | 1                  | 1                  |


    -   Add your observations on the performance of each model on the chosen dataset.

    | ML Model Name            | Observation about model performance                                                                                                                                      |
    | :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | Logistic Regression      | Performed very well with high scores across all metrics, indicating strong predictive power and good generalization.                                                     |
    | Decision Tree            | Achieved perfect scores (1.0) for all metrics. Further validation on new dataset is suitable as test data was obtained from same data set.                               |
    | kNN                      | Showed the lowest performance among all models, indicating it may not be the most suitable model for this dataset without significant hyperparameter tuning.             |
    | Naive Bayes              | Delivered strong performance, particularly with perfect recall, suggesting it's excellent at identifying all viral content, though with a slight trade-off in precision. |
    | Random Forest (Ensemble) | Achieved perfect scores (1.0) for all metrics, which is very strong performance but might indicate overfitting. Further validation like Decision Tree may be useful.     |
    | XGBoost (Ensemble)       | Also achieved perfect scores (1.0) for all metrics, which is great performance. But similarly may require additional validation with new data.                           |
