import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from scipy import stats
from statsmodels.stats.multitest import multipletests
import networkx as nx
import glob
import sspa
import numbers
import os.path



# download pathway data
reactome_pathways_file = glob.glob('Reactome_Homo_sapiens_pathways_ChEBI_R*')
if reactome_pathways_file:
    reactome_pathways_file = reactome_pathways_file[0]
    pass
else:
    reactome_pathways = sspa.process_reactome(organism="Homo sapiens", download_latest=True, filepath='.')
    reactome_pathways_file = glob.glob('Reactome_Homo_sapiens_pathways_ChEBI_R*')[0]


class MTBLSDataset:
    '''
    Class representing a Metabolights dataset.

    Parameters:
    - file_path (str): The path to the directory containing maf file and sample metadata 
    - id (str): The MetaboLights identifier of the dataset.
    - node_name (str): node name, can be the same as id
    - md_group (str): The metadata column within the sample metadata file.
    - md_filter (dict): Dictionary specifying which metadata label is case and which is the control (e.g. {'Case':'Severe','Control': 'Mild'})
    - identifier (str, optional): The identifier column name. Defaults to 'database_identifier'.
    - remove_suffix (int, optional): The number of characters to remove from the end of the sample names. Defaults to None.
    - remove_prefix (int, optional): The number of characters to remove from the beginning of the sample names. Defaults to None.
    - outliers (list, optional): The list of sample IDs to be removed as outliers. Defaults to None.
    - pathway_level (bool, optional): Whether to perform analysis at the pathway level. Defaults to False.

    Attributes:
    - raw_data (DataFrame): The raw data from the dataset.
    - compound_mappers (None): Placeholder for DataFrame of alternative identifiers.
    - processed_data (DataFrame): The processed data from the dataset.
    - DA_metabolites (None): Placeholder for DA metabolites DataFrame containing results of DA testing.
    - pathway_data (None): Placeholder for pathway transformed data matrix.
    - pathway_coverage (None): Placeholder for pathway coverage dictionary

    Methods:
    - read_data(): Reads the data from the dataset file.
    - preprocess_data(): Preprocesses the raw data.
    - get_pathway_data(): ssPA transformation 
    - plot_qc(): Plots PCA and box plots
    - da_testing(): Performs differential analysis testing.
    '''

    def __init__(self, file_path, id, node_name, md_group, md_filter, identifier='database_identifier', remove_suffix=None, remove_prefix=None,  outliers=None, pathway_level=False):
        self.file_path = file_path
        self.raw_data = None
        self.remove_suffix = remove_suffix
        self.remove_prefix = remove_prefix
        self.compound_mappers = None
        self.processed_data = None
        self.metadata = None
        self.id = id
        self.node_name = node_name
        self.md_group = md_group
        self.md_filter = md_filter
        self.DA_metabolites = None
        self.identifier = identifier
        self.outliers = outliers
        self.pathway_data = None
        self.pathway_level = pathway_level
        self.pathway_coverage = None

        self.read_data()
        # self.preprocess_data()
        self.get_pathway_data()
        self.da_testing()
        

    def read_data(self):
        '''
        Reads the data from the dataset file and sends it for preprocessing.

        Returns:
        - processed_data (DataFrame): The processed data from the dataset.
        - metadata (DataFrame): The metadata from the dataset.
        '''
        # check if there are multiple files in folder
        metadata = pd.read_csv(self.file_path + '/s_' + self.id + '.txt', sep = '\t', encoding='unicode_escape')

        metadata['Sample Name'] = metadata['Sample Name'].astype(str)
        if self.remove_suffix:
            metadata['Sample Name'] = metadata['Sample Name'].str[:-self.remove_suffix]
        if self.remove_prefix:
            metadata['Sample Name'] = metadata['Sample Name'].str[self.remove_prefix:]

        self.metadata = metadata

        files = glob.glob(self.file_path + '/*' + 'maf.tsv')
        if len(files) > 1:
            print(len(files), 'assay files found')
            # read in all files and concatenate
            dfs = []
            dfs_proc = []
            for f in files:
                df = pd.read_csv(f, sep='\t')
                dfs.append(df)
                # df.index = df['Samples']
                self.raw_data = df

                df_proc = self.preprocess_data()
                dfs_proc.append(df_proc)

            self.raw_data = pd.concat(dfs, axis=1)
            
            # inner join removes samples not present in all assays
            proc_data = pd.concat(dfs_proc, axis=1, join='inner')
            # aveage same columns
            proc_data = proc_data.groupby(by=proc_data.columns, axis=1).apply(lambda g: g.mean(axis=1) if isinstance(g.iloc[0,0], numbers.Number) else g.iloc[:,0])
            # remove duplicate 'Group' columns
            proc_data = proc_data.loc[:, ~proc_data.columns.duplicated()]

            # move group column to end
            cols = [col for col in proc_data.columns if col != 'Group'] + ['Group']
            self.processed_data = proc_data[cols]

        else:
            print(len(files), 'assay file found')
            data = pd.read_csv(files[0], sep='\t')
            # data.index = data['Samples']
            self.raw_data = data

            proc_data = self.preprocess_data()
            proc_data = proc_data.groupby(by=proc_data.columns, axis=1).apply(lambda g: g.mean(axis=1) if isinstance(g.iloc[0,0], numbers.Number) else g.iloc[:,0])
            self.processed_data = proc_data

        return self.processed_data, self.metadata


    def preprocess_data(self):
        '''
        Preprocesses the raw data (filtering, imputation, scaling).

        Returns:
        - data_scaled (DataFrame): The preprocessed and scaled data.
        '''
        data_filt = self.raw_data.copy()

        # repalce decimal in mz ratios
        try:
            data_filt['mass_to_charge'] = data_filt['mass_to_charge'].round(2)
            data_filt['mass_to_charge'] = data_filt['mass_to_charge'].astype('str').apply(lambda x: re.sub(r'\.', '_', x))
        except KeyError:
            pass

        self.all_ids = data_filt.iloc[:, ~data_filt.columns.isin(self.metadata['Sample Name'].tolist())]

        # # make a new identifier colum from chebi and metabolite_identification, prioritise chebi
        # data_filt['Identifier'] = data_filt['database_identifier'].replace('unknown', np.NaN)
        
        # try:
        #     data_filt['Identifier_filled'] = data_filt['Identifier'].fillna(data_filt['mass_to_charge'])
        #     data_filt = data_filt[data_filt['Identifier_filled'].notna()]
        #     data_filt.index = data_filt['Identifier_filled']
        # except KeyError:
        #      data_filt.index = data_filt['Identifier']

        # # set chebi as index
        data_filt = data_filt[data_filt[self.identifier].notna()]
        # drop 'unknown'
        # if there are no chebis in the whole assay file, drop it
        if data_filt.shape[0] == 0:
            print('No CHEBIS for assay')
            return None
        else:
            data_filt = data_filt[data_filt[self.identifier] != 'unknown']
            data_filt.index = data_filt[self.identifier]

            # remove assay specific sample suffixes
            if self.remove_suffix:
                data_filt.columns = data_filt.columns.str[:-self.remove_suffix]

            # keep only abundance data filtering on samples
            # store alternative identifiers in a dict
            samples = self.metadata['Sample Name'].tolist()
            ids = data_filt.iloc[:, ~data_filt.columns.isin(samples)]
            self.id_dict = ids.to_dict()
            data_filt = data_filt.iloc[:, data_filt.columns.isin(samples)]


            # ensure all data is numeric
            data_filt = data_filt.apply(pd.to_numeric, errors='coerce')

            # Transpose
            data_filt = data_filt.T

            # There will be QC samples so better filter on metadata at this point
            md_dict = dict(zip(self.metadata['Sample Name'], self.metadata[self.md_group]))
            # add metadata column
            data_filt['Group'] = data_filt.index.map(md_dict)

            # filter on metadata
            data_filt = data_filt[data_filt['Group'].isin(self.md_filter.values())]
            data_filt = data_filt.drop(columns=['Group'])
            # drop outliers
            if self.outliers:
                data_filt = data_filt.drop(self.outliers)

            # Missingness checks 
            # replace empty strings with NaN
            data_filt = data_filt.replace(['', ' '], np.nan)
            # Delete colums and rows where all values are missing
            data_filt = data_filt.dropna(axis=0, how='all')
            data_filt = data_filt.dropna(axis=1, how='all')

            # Delete rows and columns where all values are 0 
            data_filt = data_filt.loc[:, (data_filt != 0).any(axis=0)]
            data_filt = data_filt.loc[(data_filt != 0).any(axis=1), :]

            data_filt = data_filt.dropna(axis=1, thresh=0.5*data_filt.shape[0])
            missing_pct = data_filt.isnull().sum().sum() / (data_filt.shape[0] * data_filt.shape[1]) * 100
            print(f"Missingness: {missing_pct:.2f}%")

            # impute missing values
            imputer = KNNImputer(n_neighbors=2, weights="uniform").set_output(transform="pandas")
            data_imputed = imputer.fit_transform(data_filt)

            # log transformation
            data_imputed = np.log(data_imputed + 1)

            # standardize
            scaler = StandardScaler().set_output(transform="pandas")
            data_scaled = scaler.fit_transform(data_imputed)

            data_scaled['Group'] = data_scaled.index.map(md_dict)
            self.processed_data = data_scaled

            return data_scaled
    
    def get_pathway_data(self):
        '''
        Performs ssPA pathway transformation on the data.
        '''
        reactome_paths = sspa.process_gmt(infile=reactome_pathways_file)
        reactome_dict = sspa.utils.pathwaydf_to_dict(reactome_paths)
        # remove CHEBI: from column names
        data = self.processed_data
        data.columns = data.columns.str.removeprefix("CHEBI:")

        # store pathway coverage stats
        cvrg_dict = {k: len(set(data.columns).intersection(set(v))) for k, v in reactome_dict.items()}
        self.pathway_coverage = cvrg_dict

        scores = sspa.sspa_KPCA(reactome_paths).fit_transform(data.iloc[:, :-1])
        scores['Group'] = self.processed_data['Group']
        self.pathway_data = scores
    
    def plot_qc(self):
        '''
        Plots PCA and boxplots, returns figure
        '''
        # PCA biplot
        pca = PCA(n_components=2).set_output(transform="pandas")
        pca_result = pca.fit_transform(self.processed_data.iloc[:, :-1])
        self.pca = pca_result

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.scatterplot(
            x=pca_result.iloc[:, 0], y=pca_result.iloc[:, 1],
            hue="Group",
            data=self.processed_data,
            alpha=0.7,
            ax=ax1
        )

        # every nth feature - display 20 features
        feature_idx = int(self.processed_data.shape[1]/20)
        filt_features = self.processed_data.iloc[:, ::feature_idx]
        filt_features['Group'] = self.processed_data['Group']
        data_long = filt_features.melt(id_vars='Group')
        sns.boxplot(data=data_long, ax=ax2, hue='Group', x='variable', y='value')
        ax2.tick_params(axis='x', rotation=90)
        ax2.axhline(0, color='red', linestyle='--')
        plt.show()

    def da_testing(self):
        '''
        Performs differential analysis testing, adds pval_df attribute containing results.
        '''
        if self.pathway_level == True:
            dat = self.pathway_data
        else:
            dat = self.processed_data

        # t-test for two groups
        case = self.md_filter['Case']
        control = self.md_filter['Control']
        
        stat, pvals = stats.ttest_ind(dat[dat['Group'] == case].iloc[:, :-1],
                        dat[dat['Group'] == control].iloc[:, :-1],
                        alternative='two-sided', nan_policy='raise')
        pval_df = pd.DataFrame(pvals, index=dat.columns[:-1], columns=['P-value'])
        pval_df['Stat'] = stat
        pval_df['Direction'] = ['Up' if x > 0 else 'Down' for x in stat]
        self.pval_df = pval_df

        # fdr correction 
        pval_df['FDR_P-value'] = multipletests(pvals, method='fdr_bh')[1]

        # return significant metabolites
        self.DA_metabolites = pval_df[pval_df['FDR_P-value'] < 0.05].index.tolist()
        print(f"Number of differentially abundant metabolites: {len(self.DA_metabolites)}") 

        # generate tuples for nx links
        self.connection = [(self.node_name, met) for met in self.DA_metabolites]
        self.full_connection = [(self.node_name, met) for met in self.processed_data.columns[:-1]]
