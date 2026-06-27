import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

df = pd.read_csv("Mall_Customers.csv")
df.columns = df.columns.str.strip()
df.drop("CustomerID", axis=1, inplace=True)
df["Genre"] = df["Genre"].map({"Male": 0, "Female": 1})
X = df[["Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)
df["Cluster"] = labels

score = silhouette_score(X, labels)
print("Silhouette Score:", score)

plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], c=labels, cmap='rainbow')
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

linked = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()