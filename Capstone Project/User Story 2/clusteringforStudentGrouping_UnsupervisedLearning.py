import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("students_data.csv") # if doing in the local then give absolute path like c://users/...

# Selecting relevant columns for clustering (numerical features)
features = data[["studytime", "freetime", "goout", "absences", "G3", "failures", "G1", "G2"]]

# Standardize the features for consistency in scale
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 1. Elbow Method for optimal 'k'
inertia_values = []
k_range = range(2, 10)  # Testing various values of k

for k in k_range:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(scaled_features)
    inertia_values.append(kmeans_model.inertia_)

# Plot the Elbow Curve to find the optimal k
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia_values, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Determining Optimal k")
plt.show()

# 2. Silhouette Score to confirm the optimal k
silhouette_vals = []

for k in k_range:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans_model.labels_)
    silhouette_vals.append(score)

# Plot Silhouette Scores to evaluate clustering quality
plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_vals, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal k")
plt.show()

# 3. Apply K-means with the selected number of clusters (assumed k=4 from the previous analysis)
optimal_k = 4
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42)
data["Cluster_Label"] = kmeans_model.fit_predict(scaled_features)

# 4. Reduce data to 2D using PCA for visualization
pca_model = PCA(n_components=2)
pca_result = pca_model.fit_transform(scaled_features)

# Add PCA components to the dataframe for easy plotting
data["PCA_Component1"] = pca_result[:, 0]
data["PCA_Component2"] = pca_result[:, 1]

# Plot the 2D PCA visualization of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PCA_Component1", y="PCA_Component2", hue="Cluster_Label", data=data, palette="viridis", s=100)
plt.title("2D PCA Visualization of Clusters (k=4)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# 5. Reduce data to 3D using PCA for a more detailed visualization
pca_model_3d = PCA(n_components=3)
pca_result_3d = pca_model_3d.fit_transform(scaled_features)

# Add the third PCA component to the dataframe
data["PCA_Component1"] = pca_result_3d[:, 0]
data["PCA_Component2"] = pca_result_3d[:, 1]
data["PCA_Component3"] = pca_result_3d[:, 2]

# Plot the 3D PCA visualization of clusters
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
scatter_plot = ax.scatter(data["PCA_Component1"], data["PCA_Component2"], data["PCA_Component3"], c=data["Cluster_Label"], cmap="viridis", s=100)

# Add labels and title
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.title("3D PCA Visualization of Clusters (k=4)")

# Add a color legend for clarity
legend = ax.legend(*scatter_plot.legend_elements(), title="Cluster")
ax.add_artist(legend)

plt.show()
