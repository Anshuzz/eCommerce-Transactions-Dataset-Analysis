from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA

# Prepare data for clustering
# Aggregate customer transaction and profile data
customer_data = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'Price_x': 'mean'
}).reset_index()

# Add customer profile features
customer_data = pd.merge(customer_data, customers[['CustomerID', 'Region']], on='CustomerID', how='left')

# Encode categorical variables
customer_data_encoded = pd.get_dummies(customer_data, columns=['Region'], drop_first=True)

# Normalize the data
scaler = StandardScaler()
normalized_customer_data = scaler.fit_transform(customer_data_encoded.iloc[:, 1:])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(normalized_customer_data)

# Add cluster labels to the data
customer_data['Cluster'] = clusters

# Evaluate clustering using Davies-Bouldin Index
db_index = davies_bouldin_score(normalized_customer_data, clusters)
print(f"Davies-Bouldin Index: {db_index}")

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_customer_data)

customer_data['PCA1'] = pca_result[:, 0]
customer_data['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster', data=customer_data, palette='Set2', s=100
)
plt.title('Customer Segments')
plt.show()

# Save clustering results
customer_data.to_csv('Customer_Segments.csv', index=False)
print("Customer segmentation results saved to 'Customer_Segments.csv'")
