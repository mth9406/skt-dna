import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import networkx as nx

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, 
                 enb_num,
                 preds, 
                 labels,
                 graphs, 
                 num_obs,
                 cache= None 
                 ):
        
        self.enb_num = enb_num
        self.graphs = graphs # graph-files: list of pandas DataFrames
        self.preds = preds.iloc[:num_obs, :]
        self.labels = labels.iloc[:num_obs, :]
        self.x = np.arange(num_obs)
        if cache is None:    
            min = np.minimum(labels.min(), preds.min())
            max = np.maximum(labels.max(), preds.max())
        else: 
            columns = list(preds.columns)
            min = cache['min'][columns]
            max = cache['max'][columns]
        pad = (max-min)/10
        self.y_lb = min - pad # lower-bound
        self.y_ub = max + pad # upper-bound
        columns = list(labels.columns)
        num_ts = len(columns)
        self.columns = columns
        self.num_obs = num_obs
        self.num_ts = num_ts
        self.options = {
                        'node_color': 'skyblue',
                        'node_size': 500,
                        'width': 0.5 ,
                        'arrowstyle': '-|>',
                        'arrowsize': 20,
                        'alpha' : 1,
                        'font_size' : 8
                    }
        fig = plt.figure(figsize= (5,5+5*num_ts))
        gs1 = GridSpec(5+5*num_ts, 5)
        fig.add_subplot(gs1[:5, :]) # a grid for a graph 
        for i in range(5,5+5*num_ts,5):
            fig.add_subplot(gs1[i:i+5, :]) # a grid for a time series
        # graph 
        fig.axes[0].axis('off')
        fig.axes[0].set_title(f'relation graph enb{enb_num}')
        self.fig = fig
        self.pos = None

        for i, col in enumerate(columns):
            setattr(self, f'labels_{col}', Line2D([], [], color= 'black', label= f'label: {col}'))
            setattr(self, f'preds_{col}', Line2D([], [], color= 'red', label= f'preiction: {col}'))
            
            fig.axes[i+1].add_line(getattr(self, f'labels_{col}'))
            fig.axes[i+1].add_line(getattr(self, f'preds_{col}'))
            # fig.axes[i].plot(df[col], label= 'imputed', alpha= 0.2, c= 'red')
            fig.axes[i+1].legend()
            fig.axes[i+1].set_ylim(self.y_lb.iloc[i], self.y_ub.iloc[i])
            fig.axes[i+1].set_xlim(0, num_obs)
            fig.axes[i+1].set_title(f'{col}')
            # fig.axes[i+1].set_xlabel('x')
        plt.tight_layout()
        animation.TimedAnimation.__init__(self, fig, interval = 1000, blit=True)

    def _draw_frame(self, framedata):
        i = framedata

        self.fig.axes[0].cla()
        G = nx.from_pandas_adjacency(self.graphs[i], create_using=nx.DiGraph)
        G = nx.DiGraph(G)
        if self.pos is None:
            self.pos = nx.circular_layout(G)

        nx.draw(G, with_labels=True, ax= self.fig.axes[0], pos= self.pos, **self.options)

        for j, col in enumerate(self.columns):
            getattr(self, f'labels_{col}').set_data(self.x[:i], self.labels.iloc[:i, j])
            getattr(self, f'preds_{col}').set_data(self.x[:i], self.preds.iloc[:i, j])
        
        self._drawn_artists = []
        for col in self.columns: 
            self._drawn_artists.append(getattr(self, f'labels_{col}'))
            self._drawn_artists.append(getattr(self, f'preds_{col}'))

    def new_frame_seq(self):
        return iter(range(self.num_obs))

    def _init_draw(self):
        lines = []
        for col in self.columns: 
            lines.append(getattr(self, f'labels_{col}'))
            lines.append(getattr(self, f'preds_{col}'))
        for l in lines:
            l.set_data([], [])