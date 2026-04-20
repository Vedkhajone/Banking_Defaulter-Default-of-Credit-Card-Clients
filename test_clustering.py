from src.preprocessing import full_preprocessing
from src.clustering import perform_clustering, attach_clusters, analyze_clusters

X, y, df = full_preprocessing("data/raw/data.csv")

clusters, kmeans = perform_clustering(X)

df_clustered = attach_clusters(df, clusters)

analyze_clusters(df_clustered)

from src.insights import cluster_risk_analysis, business_insights

cluster_risk_analysis(df_clustered)
business_insights(df_clustered)



import os
import joblib

# Create models folder
os.makedirs("models", exist_ok=True)

# Save kmeans model
joblib.dump(kmeans, "models/kmeans.pkl")

print("✅ KMeans model saved!")