from Data import Data
class AggregatedData(Data):
    def __init__(self,generator,connection):
        super().__init__(generator,connection)
        
    def make_df(self):
        link = self.generator.generate_JSON_link()
        doi = ['avg','max','min','stddev']
        all_data_df = pd.DataFrame(columns = doi+['ts'])
        data, nextlink = self.get_dict_from_JSON(link)
        all_data_df = all_data_df.append(self.get_df(data),ignore_index=True)
        while nextlink is not None:
            data, nextlink = self.get_dict_from_JSON(nextlink)
            all_data_df = all_data_df.append(self.get_df(data),ignore_index=True)
        #all_data_df.set_index('ts', inplace=True)      
        self.df = all_data_df
        self.dfs = [self.df]
            
    
    def get_dict_from_JSON(self,link):
        response = urllib.request.urlopen(link)
        data = json.loads(response.read())
        try:
            nextlink = data['@iot.nextLink']
        except KeyError: 
            nextlink = None
        return data, nextlink
        
    def get_df(self,data):
        cols = config.doi+['ts']
        df = pd.DataFrame(columns = cols)
        for i in range(len(data['value'])):
            ts = datetime.strptime(data['value'][i]['resultTime'],config.dateformat_ymdhms)
            data_points = [data['value'][i]['result'][dp] for dp in config.doi]
            df=df.append(pd.DataFrame([data_points+[ts]],columns=cols),ignore_index=True)
        return df
        
    def plot_df(self,y='max',xbase=1000.,ybase=40.):
        fig, ax = plt.subplots()
        ax.plot(self.df.index,self.df[y],linewidth=0.1)
        xloc = plticker.MultipleLocator(base=xbase) # this locator puts ticks at regular intervals
        yloc = plticker.MultipleLocator(base=ybase)
        ax.xaxis.set_major_locator(xloc)
        ax.yaxis.set_major_locator(yloc)
        plt.show()
        
    def preprocess(self, method = 'mean'):
        for i in range(len(self.dfs)):
            print(self.dfs[i].columns)
            if method == 'mean': # standardize
                normalized_df=(self.dfs[i].drop(['ts'],axis=1)-self.dfs[i].drop(['ts'],axis=1).mean())/self.dfs[i].drop(['ts'],axis=1).std()
                normalized_df['ts'] = self.dfs[i]['ts']  
                self.dfs[i] = normalized_df
            elif method == 'min-max':
                normalized_df=(self.dfs[i].drop(['ts'],axis=1)-self.dfs[i].drop(['ts'],axis=1).min())/(self.dfs[i].drop(['ts'],axis=1).max()-self.dfs[i].drop(['ts'],axis=1).min())
                normalized_df['ts'] = self.dfs[i]['ts']
                self.dfs[i] = normalized_df
            else: 
                print('No preprocessing scheme specified')
