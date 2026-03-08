# Display leaderboard for a dataset and fold
genbio-leaderboard leaderboard --dataset RNA/translation-efficiency-muscle --fold 0

# Display submission history for a specific user
genbio-leaderboard history --dataset RNA/translation-efficiency-muscle --fold 0 --user caleb.ellington

# Export benchmark data to CSV
genbio-leaderboard export --output benchmark_data.csv