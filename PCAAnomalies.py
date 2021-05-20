'''
Child class to AnomalyModel. Inherits X which is a np matrix of features(normalized) x samples as well as labels
'''
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import config

import pickle

from AnomalyModel import AnomalyModel

class PCAAnomalies(AnomalyModel):
    def __init__(self,anomaly_dict, settings, features, n_components = 0.99):
        super().__init__(anomaly_dict, settings, features)
        self.pca = PCA(
            n_components = n_components,
            svd_solver = 'full')
        self.end_indices = np.array([[int(x.end_index) for x in list(anomaly_dict.values())]])
        self.df_numbers = np.array([[int(x.df_number) for x in list(anomaly_dict.values())]])
        self.irl_labels = np.array([[int(x.irl_label) for x in list(anomaly_dict.values())]])
        self.df = pd.DataFrame(
            np.append(self.X, np.append(self.end_indices,np.append(self.df_numbers,self.irl_labels,axis=0),axis=0),axis=0).transpose(),
            columns=self.feature_labels+['end index','df_number','irl_label'])
        self.feature = anomaly_dict[0].feature # Only works when there is only one feature
        
        
    def fit_PCA(self):
        print(f"Dimension of X: {np.size(self.X)}")
        self.pca.fit_transform(self.X)
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Singular values: {self.pca.singular_values_}")
        
    def save_pca(self,name):
        with open(self.get_pca_name(name),'wb') as f:
            pickle.dump(self.pca, f) 
            
    def load_pca(self,name):
        self.df = pickle.load(
            open(self.get_pca_name(name),'rb')
            )   
    
    def get_pca_name(self,name):
        return f"{config.pca_path}{name}_pca.json"
        
    def get_cov(self):
        self.cov = self.pca.get_covariance()
        print(self.cov)
        
    def plot_components(self,features):
        fig, axs = plt.subplots(len(features), figsize = config.figsize)
        for i, feature in enumerate(features):
            cmap_feature = self.df[feature]
            sc = axs[i].scatter(
                self.pca.components_[0,:],
                self.pca.components_[1,:],
                s = 1,
                c = cmap_feature,
                cmap = 'viridis',
                norm = mpl.colors.Normalize(vmin=min(cmap_feature), vmax=max(cmap_feature)))
            axs[i].set_xlabel('First principal axis')
            axs[i].set_ylabel('Second principal axis')
            axs[i].set_title(feature)
            cbar = fig.colorbar(sc,ax=axs[i])
            cbar.set_label(config.anomaly_units[feature])
        fig.tight_layout() 
        plt.suptitle(f"{self.feature}")
        plt.show()
        
    def plot_components_3d(self):
        feature = 'Duration'
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        cmap_feature = self.df[feature]
        try:
            cbar_label = config.anomaly_units[feature]
        except KeyError:
            cbar_label = ''
        sc = ax.scatter(
            self.pca.components_[0,:],
            self.pca.components_[1,:],
            self.pca.components_[2,:],
            c = cmap_feature,
            cmap = 'viridis',)
        ax.set_xlabel('First principal axis')
        ax.set_ylabel('Second principal axis')
        ax.set_zlabel('Third principal axis')
        cbar = fig.colorbar(sc,ax=ax)
        cbar.set_label(config.anomaly_units[feature])
        plt.suptitle(f"{self.feature}")
        plt.show()
        
    def plot_components_log(self):
        cmap_feature = self.df['Duration']
        sc = plt.scatter(
            np.log(self.pca.components_[0,:]),
            np.log(self.pca.components_[1,:]),
            c = cmap_feature,
            cmap = 'viridis',
            norm = mpl.colors.LogNorm(vmin = min(cmap_feature), vmax = max(cmap_feature)))
        plt.xlabel('First principal axis')
        plt.ylabel('Second principal axis')
        plt.colorbar(sc)
        plt.show()
        
    def plot_hist_pca(self,component = 1):
        component -=1
        colors = config.colors
        for i, color in enumerate(colors):
            j = (self.df['labels'] == i).to_numpy() 
            plt.hist(
                self.pca.components_[component,:][j],
                bins = 50,
                color = color,
                alpha = 0.5,
                edgecolor = 'k',
                stacked = True,
                #label = f"Category {i}",
                )
        plt.title(f"PCA component {component+1} categories according to Kmeans")
        #plt.legend()
        plt.show()
        
    def scree_plot(self):
        kaisers_rule = 1
        fig, axs = plt.subplots(2, figsize = config.figsize)
        axs[0].plot(
            np.arange(1,len(self.pca.explained_variance_ratio_)+1),
            self.pca.explained_variance_ratio_,
            linestyle = '-', marker = 'o', color = 'b',label = 'Explained variance per PC')
        axs[0].plot(
            np.arange(1,len(self.pca.explained_variance_ratio_)+1),
            np.cumsum(self.pca.explained_variance_ratio_),
            linestyle = '-', marker = 'o', color = 'g',label = 'Cumulated explained variance')
        axs[0].set_xlabel('Principal component #')
        axs[0].set_ylabel('Explained variance ratio')
        axs[0].set_title('Explained variance per PC')
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(
            np.arange(1,len(self.pca.singular_values_)+1),
            self.pca.singular_values_,
            linestyle = '-', marker = 'o', color = 'b',label = 'Singular values')
        axs[1].plot(
            [1,len(self.pca.singular_values_)],
            [kaisers_rule,kaisers_rule],
            linestyle = '--', color = 'r',label = 'Kaiser\'s rule')
        axs[1].set_xlabel('Principal component #')
        axs[1].set_ylabel('Eigenvalues')
        axs[1].set_title('Eigenvalues per PC')
        axs[1].legend()
        axs[1].grid(True)
        plt.suptitle(f"{self.feature}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
                
    def get_argmax(self,col='Duration'):
        print(self.df.iloc[self.df[col].argmax()]['df_number'])
        return self.df[col].argmax(), int(self.df.iloc[self.df[col].argmax()]['df_number'])
        
    def set_labels(self,labels):
        self.df['labels'] = labels
        
    def plot_components_labels(self,n_categories,plot_irl_observation=False):
        fig, axs = plt.subplots(2, figsize = config.figsize)
        cmap = plt.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(0, n_categories, n_categories+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        if plot_irl_observation:
            markers = ['o','o']
        else:
            markers = ['o']
        linewidths = [0,2]
        for k,m in enumerate(markers):
            i = (self.df['irl_label'] == k).to_numpy()   
            axs[0].scatter(
                self.pca.components_[0,i],
                self.pca.components_[1,i],
                c = self.df['labels'][i],
                s = 10+10*k,
                marker = m,
                cmap = cmap,
                norm = norm,
                linewidths = linewidths[k],
                edgecolors = 'k')
            #axs[i].set_title(feature)
            axs[1].scatter(
                self.pca.components_[0,i],
                self.pca.components_[2,i],
                c = self.df['labels'][i],
                s = 10+10*k,
                marker = m,
                cmap = cmap,
                norm = norm,
                linewidths = linewidths[k],
                edgecolors = 'k')
            
        axs[0].set_xlabel('First principal axis')
        axs[0].set_ylabel('Second principal axis')
        axs[1].set_xlabel('First principal axis')
        axs[1].set_ylabel('Third principal axis')
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[1])
        plt.suptitle(f"{self.feature}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.show()
        
        
class KPCAAnomalies(PCAAnomalies):
    def __init__(self,anomaly_dict, n_components = 0.99, gamma = 10):
        super().__init__(anomaly_dict, n_components = 0.99)
        self.pca = KernelPCA(kernel = 'rbf', fit_inverse_transform = True, gamma = gamma)

    def fit_PCA(self):
        print(f"Dimension of X: {np.size(self.X)}")
        self.kpca_transform = self.pca.fit_transform(self.X)
        explained_variance = np.var(self.kpca_transform, axis=0) # kpca doesnt have explained variance 
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        print(f"Explained variance ratio: {explained_variance_ratio}")
        #np.cumsum(explained_variance_ratio)
        
    def plot_components(self,features = ['Duration','rms','frequency']):
        fig, axs = plt.subplots(len(features), figsize = config.figsize)
        for i, feature in enumerate(features):
            cmap_feature = self.df[feature]
            sc = axs[i].scatter(
                self.kpca_transform[0,:],
                self.kpca_transform[1,:],
                s = 1,
                c = cmap_feature,
                cmap = 'viridis',
                norm = mpl.colors.Normalize(vmin=min(cmap_feature), vmax=max(cmap_feature)))
            axs[i].set_xlabel('First principal axis')
            axs[i].set_ylabel('Second principal axis')
            axs[i].set_title(feature)
            cbar = fig.colorbar(sc,ax=axs[i])
            cbar.set_label(config.anomaly_units[feature])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.show()

               
        

        
   
        
