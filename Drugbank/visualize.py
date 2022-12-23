import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering

z = np.load("drug_embedding.npy")

z_embed = TSNE(n_components=2, learning_rate='auto', init='random', 
            perplexity=3).fit_transform(z)
plt.scatter(z_embed[:, 0], z_embed[:, 1])
plt.axis("off")
plt.savefig("tnse.png", bbox_inches = "tight", transparent = True)