import pandas as pd
from job_matcher import calculate_fit

# 1. Create your labeled test dataset
#    - target = 1 means it's a good match.
#    - target = 0 means it's a bad match.
data = {
    'resume_skills': [
        ['python', 'pandas', 'scikit-learn', 'tensorflow'],
        ['react', 'javascript', 'css', 'html'],
        ['java', 'spring boot', 'mysql'],
        ['aws', 'docker', 'kubernetes', 'terraform'],
        ['python', 'flask', 'sql']
    ],
    'job_skills': [
        ['python', 'tensorflow', 'machine learning', 'data analysis'], # Good match
        ['react', 'javascript', 'css', 'nodejs'],                     # Good match
        ['python', 'django', 'postgresql'],                            # Bad match
        ['aws', 'docker', 'ci/cd'],                                    # Good match
        ['java', 'spring', 'nosql']                                    # Bad match
    ],
    'target': [1, 1, 0, 1, 0]
}
test_df = pd.DataFrame(data)

# 2. Define a threshold for what your model considers a "match"
#    This is the fit score above which you classify a pair as a good match.
#    You can adjust this value to see how it affects your accuracy.
FIT_SCORE_THRESHOLD = 50

# 3. Run the predictions
def predict(row):
    score = calculate_fit(row['resume_skills'], row['job_skills'])
    # If the score is above the threshold, predict 1 (good match), else 0
    return 1 if score >= FIT_SCORE_THRESHOLD else 0

test_df['prediction'] = test_df.apply(predict, axis=1)

# 4. Calculate the accuracy
correct_predictions = (test_df['prediction'] == test_df['target']).sum()
total_predictions = len(test_df)
accuracy = (correct_predictions / total_predictions) * 100

# 5. Print the results
print(f"Fit Score Threshold: {FIT_SCORE_THRESHOLD}%")
print("-" * 30)
print(test_df)
print("-" * 30)
print(f"Correct Predictions: {correct_predictions}")
print(f"Total Predictions:   {total_predictions}")
print(f"Model Accuracy:      {accuracy:.2f}%")