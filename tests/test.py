import genbio.leaderboard as gl

# Select a dataset and fold, and provide a username
task = gl.BenchmarkTask(name='RNA/translation-efficiency-muscle', fold='0', user='caleb.ellington')

# Print dataset description and documentation
task.describe()

# Load the training and test data
train_df, test_df = task.setup()

# Build your model, make predictions
mean_pred = train_df['labels'].mean()
train_pred_df = train_df.copy()
train_pred_df['labels'] = mean_pred
test_pred_df = test_df.copy()
test_pred_df['labels'] = mean_pred

# Compute intermediate train metrics
task.evaluate(train_pred_df, train_df)

# Make a submission
task.submit(test_pred_df, name='mean_predictor_v0.0', description='Dummy submission using mean prediction')