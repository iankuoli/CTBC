import networkx as nx
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
import pickle
from tqdm import tqdm_notebook
from multiprocessing import Pool
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
warnings.filterwarnings("ignore")



#
# Parameter Settings
# --------------------------------------------------------------------------------------------------------------------------

# Model settings
latent_dim = 300
epochs = 600
lr = 100
lambda_V = 1e-5
lambda_w = 1e-4
param_alpha = 1
focal_alpha = 0.1
focal_gamma = 1.5
edge_attrs = ['dim_1', 'dim_2', 'dim_3']
L = 20

# Graph fetching settings
weight_type = 'embed'  # [weight_none, weight_log, count_larger8000]
offset = 0



CustInfo = pd.read_csv('../AMLAML/customerinformation.csv')
CustInfo['open_date'] = pd.to_datetime(CustInfo.Open_Date)

SARCase = pd.read_csv('../AMLAML/sarcase.csv')
SARCase['created_date'] = pd.to_datetime(SARCase.Created_Date)

WireTrans = pd.read_csv('data/new_wire.csv')
WireTrans['trans_date'] = pd.to_datetime(WireTrans.trandt)
WireTrans.head()


'''
### Aggregation of values
In transaction data, we want to aggregate transaction values in the past few months by using groupby() and sum() functions.
'''



# Convert to undirected vertex index 
def group_trans_by_sum_weight(df, col_from_vertex='org_cust', col_to_vertex='bnf_cust'):
    df['vertex_index'] = df.apply(lambda x: str(set([x[col_from_vertex], x[col_to_vertex]])), axis=1)
    df_new = df.groupby('vertex_index').sum().reset_index()
    df_new[col_from_vertex] = df_new.apply(lambda x: x['vertex_index'][1:-1].split(',')[0][1:-1], axis=1)
    df_new[col_to_vertex] = df_new.apply(lambda x: x['vertex_index'][1:-1].split(',')[1][2:-1], axis=1)
    return df_new



# Convert to undirected vertex index 
def group_trans_by_count_larger(df, thres=8000, 
                                col_from_vertex='org_cust', col_to_vertex='bnf_cust', 
                                col_wire_in='WIRE_AMTIN', col_wire_out='WIRE_AMTOT'):
    df['vertex_index'] = df.apply(lambda x: str(set([x[col_from_vertex], x[col_to_vertex]])), axis=1)
    df['norm_wire_amt'] = df.apply(lambda x: 1 if x[col_wire_out] > thres or x[col_wire_in] > thres else 0, axis=1)
    df_new = df.groupby('vertex_index').sum().reset_index()
    df_new[col_from_vertex] = df_new.apply(lambda x: x['vertex_index'][1:-1].split(',')[0][1:-1], axis=1)
    df_new[col_to_vertex] = df_new.apply(lambda x: x['vertex_index'][1:-1].split(',')[1][2:-1], axis=1)
    return df_new



## Fetch data by date



class DateYM:
    def __init__(self, year, month):
        self.year = year
        self.month = month - 1    # 0 ~ 11, from Jan to Dec
        
    def export_tuple(self):
        return (self.year, self.month+1)
    
    def add_year(self, y):
        self.year += y
        
    def substract_year(self, y):
        self.year -= y
        
    def add_month(self, m):
        self.month += m
        self.year += math.floor(self.month / 12)
        self.month = self.month % 12
        
    def subtract_month(self, m):
        self.month -= m
        tmp_year = math.floor(self.month / 12)
        self.year += tmp_year
        self.month += -tmp_year * 12
        
    def is_larger_than(self, ym):
        return self.year*12 + self.month > ym.year*12 + ym.month
    
    def is_smaller_than(slef, ym):
        return self.year*12 + self.month < ym.year*12 + ym.month
    
    def is_equal(self, ym):
        return self.year*12 + self.month == ym.year*12 + ym.month

    
def list_date_tuples(from_date, to_date):
    ret = []
    tmp_date = DateYM(*from_date.export_tuple())
    while not tmp_date.is_larger_than(to_date):
        ret.append(tmp_date.export_tuple())
        tmp_date.add_month(1)
    return ret


def fetch_data_by_month(date_ym, trans_offset=6):
    year, month = date_ym
    
    # Get view: WireTrans
    from_date = pd.to_datetime("{}/{}/{}".format(month, 1, year))
    to_date = from_date + pd.DateOffset(months=1)
    offset_date = from_date - pd.DateOffset(months=trans_offset)
    view_wiretrans = WireTrans[(WireTrans.trans_date > offset_date) & 
                               (WireTrans.trans_date < to_date)]
    
    # Get view: CustInfo
    view_customer = CustInfo[CustInfo.open_date < to_date]
    
    # Attach label onto CustInfo
    target_list = SARCase[(SARCase.Status_SAR == 4) & 
                          (SARCase.created_date > from_date) & 
                          (SARCase.created_date < to_date)]['customerno'].unique()
    # print ('# of SAR customers: {}'.format(len(target_list)))
    view_customer['label'] = view_customer.apply(lambda x: 1 if x['customerno'] in target_list else 0, axis=1)
    
    return view_wiretrans, view_customer



from_date_ym = DateYM(2018, 1)
to_date_ym = DateYM(2019, 6)
list_date_seq = list_date_tuples(from_date_ym, to_date_ym)
seq_length = len(list_date_seq)
print(list_date_seq)


'''
[(2018, 1), (2018, 2), (2018, 3), (2018, 4), (2018, 5), (2018, 6), (2018, 7), (2018, 8), (2018, 9), (2018, 10), (2018, 11), (2018, 12), (2019, 1), (2019, 2), (2019, 3), (2019, 4), (2019, 5), (2019, 6)]
'''




## Embedding Projection on a Dynamic graph



def get_edge_embed(date_ym, groupby_method='sum'):
    year, month = date_ym
    df = pd.read_csv('GraphLaplacian/Edge_Attribute/VAE_Edge_attribute_{}-{}.csv'.format(year, month))
    df.columns = ['index', 'from_vertex', 'to_vertex', 'dim_1', 'dim_2', 'dim_3']
    df['vertex_index'] = df.apply(lambda x: str(set([x['from_vertex'], x['to_vertex']])), axis=1)
    if groupby_method == 'sum':
        df_new = df.groupby('vertex_index').sum().reset_index().reset_index()
    else:
        df_new = df.groupby('vertex_index').mean().reset_index().reset_index()
    df_new['from_vertex'] = df_new.apply(lambda x: x['vertex_index'][1:-1].split(',')[0][1:-1], axis=1)
    df_new['to_vertex'] = df_new.apply(lambda x: x['vertex_index'][1:-1].split(',')[1][2:-1], axis=1)
    
    return df_new




def sigmoid(x):
    #print(x)
    #if type(x) is float or int:
    #    x = -5. if x < -5. else x
    if len(x.shape) > 0:
        x[x < -5] = -5.
    else:
        x = -5. if x < -5. else x
    return 1 / (1 + np.exp(-x))




# def update_params(t):
def update_params(input_pkg):
    t, hetergraph = input_pkg
    L_t, A_ts = hetergraph
    date_ym = list_date_seq[t]
    
    key2int_t = dict_key2int[date_ym]
    int2key_t = dict_int2key[date_ym]
    y_t = groundtruth[date_ym]
    V_t = embeds[date_ym]
    w1 = classifiers['w1']
    w2 = classifiers['w2']
    we = classifiers['we']
    theta = classifiers['theta']
    
    # Construct graph
    # view_embeds = get_edge_embed(date_ym)
    # G_t = nx.from_pandas_edgelist(view_embeds, 'from_vertex', 'to_vertex', 
    #                              edge_attr=edge_attrs,
    #                              create_using=nx.MultiGraph())
    # G_t = nx.relabel_nodes(G_t, key2int_t)
    # L_t = nx.normalized_laplacian_matrix(G_t, nodelist=list(int2key_t.keys()))
    # A_t = nx.to_scipy_sparse_matrix(G_t, nodelist=int2key_t)
    # A_ts = [nx.to_scipy_sparse_matrix(G_t, weight=attr, nodelist=int2key_t) for attr in edge_attrs]
    A_t = scipy.sparse.csr_matrix(A_ts[0].shape)
    for index_attr in range(we.shape[0]):
        tmpA_t = A_ts[index_attr]
        A_t += sigmoid(we[index_attr,0]) * tmpA_t
        
    if (t + 1) < seq_length:
        date_ym_plus = list_date_seq[t + 1]
        
        key2int_t_plus = dict_key2int[date_ym_plus]
        int2key_t_plus = dict_int2key[date_ym_plus]
        index_t_2_t_plus = [key2int_t_plus[x] if x in key2int_t_plus else -1 for x in int2key_t]
        
        # Get V_t_plus aligned in V_t index order
        V_t_plus_orig = embeds[date_ym_plus]
        V_t_plus_orig = np.append(V_t_plus_orig, np.zeros((1, V_t_plus_orig.shape[1])), axis=0)
        V_t_plus = V_t_plus_orig[index_t_2_t_plus]
        
        # Get y_t_plus aligned in y_t index order
        y_t_plus_orig = groundtruth[date_ym_plus]
        y_t_plus_orig = np.append(y_t_plus_orig, np.zeros((1,1)), axis=0)
        y_t_plus = y_t_plus_orig[index_t_2_t_plus]
    else:
        V_t_plus = None
        y_t_plus = None
        key2int_t_plus = None
        int2key_t_plus = None

    if t > 0:
        date_ym_minus = list_date_seq[t - 1]
        
        key2int_t_minus = dict_key2int[date_ym_minus]
        int2key_t_minus = dict_int2key[date_ym_minus]
        index_t_2_t_minus = [key2int_t_minus[x] if x in key2int_t_minus else -1 for x in int2key_t]
        
        # Get V_t_plus aligned in V_t index order
        V_t_minus_orig = embeds[date_ym_minus]
        V_t_minus_orig = np.append(V_t_minus_orig, np.zeros((1, V_t_minus_orig.shape[1])), axis=0)
        V_t_minus = V_t_minus_orig[index_t_2_t_minus]
                
        y_t_minus_orig = groundtruth[date_ym_minus]
        y_t_minus_orig = np.append(y_t_minus_orig, np.zeros((1,1)), axis=0)
        y_t_minus = y_t_minus_orig[index_t_2_t_minus] 
    else:
        V_t_minus = None
        y_t_minus = None
        key2int_t_minus = None
        int2key_t_minus = None
    
    # Create sparse matrix S
    vec_i, vec_j, vec_v = scipy.sparse.find(A_t)
    vec_data = vec_v * sigmoid(-np.sum(V_t[vec_i, :] * V_t[vec_j, :], axis=1))
    S_t = scipy.sparse.csr_matrix((vec_data, (vec_i, vec_j)), shape=A_t.shape)
    
    # Create sparse matrix R
    l = [[u]*L for u in range(A_t.shape[0])]
    smpl_i = [item for sublist in l for item in sublist]
    l = [list(np.random.choice(A_t.shape[0], min(L*A_t[u,:].nnz, A_t.shape[0]), replace=False)) for u in range(A_t.shape[0])]
    smpl_j = [item for sublist in l for item in sublist]
    smpl_index = list(set(zip(smpl_i, smpl_j)) | set(zip(vec_i, vec_j)))
    smpl_i, smpl_j = zip(*smpl_index)
    smpl_data = sigmoid(np.sum(V_t[smpl_i, :] * V_t[smpl_j, :], axis=1))
    R_t = scipy.sparse.csr_matrix((smpl_data, (smpl_i, smpl_j)), shape=A_t.shape)
    
    # Compute normalization term
    norm_graph = 1 / (vec_data.shape[0] + smpl_data.shape[0])
    norm_laplacian = 1 / (V_t.shape[0] * V_t.shape[1])
    norm_time = 1 / (V_t.shape[0] * V_t.shape[1])
    norm_pred = 10 / V_t.shape[0]
    
    # Update variables
    for itr in range(3):
        # Create vector z_t
        vec_one = np.ones((V_t.shape[0], 1))
        tmp_recur = np.cos(theta) * V_t + np.sin(theta) * V_t_minus if V_t_minus is not None else V_t
        tmp = np.dot(tmp_recur, w1) + 0.5 * np.dot(tmp_recur * tmp_recur, w2)
        pred_t = sigmoid(tmp)
        q_t = (vec_one - y_t) * (vec_one - pred_t) + y_t * pred_t + 1e-10
        q_t[q_t > 1] = 1.
        z_t = focal_alpha * (-focal_gamma * (1 - q_t) ** (focal_gamma - 1) * np.log(q_t) + \
                             1/q_t * (1 - q_t) ** focal_gamma) * \
              (2 * y_t - vec_one) * pred_t * (vec_one - pred_t)
        
        # Compute the gradient w.r.t. V_t
        grad_V_t = norm_graph * (-param_alpha * S_t * V_t + R_t * V_t) + \
                   lambda_V * V_t + \
                   norm_laplacian * L_t * L_t * V_t
        
        if V_t_plus is not None:
            tmp_recur = np.cos(theta) * V_t_plus + np.sin(theta) * V_t
            pred_t_plus = sigmoid(np.dot(tmp_recur, w1) + 0.5 * np.dot(tmp_recur**2, w2))
            q_t_plus = (vec_one - y_t_plus) * (vec_one - pred_t_plus) + y_t_plus * pred_t_plus + 1e-10
            q_t_plus[q_t_plus > 1] = 1.
            z_t_plus = focal_alpha * (-focal_gamma * (1 - q_t_plus) ** (focal_gamma - 1) * np.log(q_t_plus) + \
                                      1/q_t_plus * (1 - q_t_plus) ** focal_gamma) * \
                       (2 * y_t_plus - vec_one) * pred_t_plus * (vec_one - pred_t_plus)
            grad_V_t += norm_time * -(V_t_plus - V_t) + \
                        norm_pred * -(np.dot(np.cos(theta) * z_t + np.sin(theta) * z_t_plus, w1.T) + 
                                      np.sin(theta) * (z_t_plus * tmp_recur * w2.T))
        
        if V_t_minus is not None:
            tmp_recur = np.cos(theta) * V_t + np.sin(theta) * V_t_minus
            grad_V_t += norm_time * (V_t - V_t_minus) + \
                        norm_pred * -(z_t * tmp_recur * w2.T)
        V_t -= lr * grad_V_t
        
        # Compute the gradient w.r.t. w1 & w2
        if V_t_minus is not None:
            tmp_recur = np.cos(theta) * V_t + np.sin(theta) * V_t_minus
            grad_w1 = -norm_pred * np.dot(tmp_recur.T, z_t) + lambda_w * w1
            w1 -= lr * grad_w1
            grad_w2 = -norm_pred * 0.5 * np.dot((tmp_recur**2).T, z_t) + lambda_w * w2
            w2 -= lr * grad_w2
            
        # Compute the gradient w.r.t. we
        grad_we = np.zeros(we.shape)
        for index_attr in range(we.shape[0]):
            tmpA_t = A_ts[index_attr]
            tmp_is, tmp_js, tmp_vs = scipy.sparse.find(tmpA_t)
            tmp = np.mean(np.log(sigmoid(np.sum(V_t[tmp_is,:] * V_t[tmp_js,:], axis=1))) * tmp_vs)
            tmp_grad_sigma_we = sigmoid(we[index_attr,0]) * (1 - sigmoid(we[index_attr,0]))
            grad_we[index_attr,0] = -param_alpha * tmp * tmp_grad_sigma_we
        we -= lr * grad_we
            
        # Compute the gradient w.r.t. theta
        grad_theta = -np.sin(theta) * (np.dot(np.dot(z_t.T, V_t), w1) + \
                                       np.dot(np.dot(z_t.T, tmp_recur * V_t), w2))
        if V_t_minus is not None:
            grad_theta = np.cos(theta) * (np.dot(np.dot(z_t.T, V_t_minus), w1) + \
                                          np.dot(np.dot(z_t.T, tmp_recur * V_t_minus), w2))
        theta -= 0.1 * lr * norm_pred * grad_theta[0][0]
        
        # Compute the gradient w.r.t. V_t_plus
        if V_t_plus is not None:
            tmp_recur = np.cos(theta) * V_t_plus + np.sin(theta) * V_t
            pred_t_plus = sigmoid(np.dot(tmp_recur, w1) + 0.5 * np.dot(tmp_recur**2, w2))
            q_t_plus = (vec_one - y_t_plus) * (vec_one * pred_t_plus) + y_t_plus * pred_t_plus + 1e-10
            q_t_plus[q_t_plus > 1] = 1.
            z_t_plus = focal_alpha * (-focal_gamma * (1 - q_t_plus) ** (focal_gamma - 1) * np.log(q_t_plus) + \
                                      1/q_t_plus + (1 - q_t_plus) ** focal_gamma) * \
                       (2 * y_t_plus - vec_one) * pred_t_plus * (vec_one - pred_t_plus)
            grad_V_t_plus = norm_graph * (-param_alpha * S_t * V_t_plus + R_t * V_t_plus) + \
                            lambda_V * V_t_plus + \
                            norm_laplacian * L_t * L_t * V_t_plus + \
                            norm_time * (V_t_plus - V_t) + \
                            norm_pred * (-z_t_plus * tmp_recur * w2.T)
            V_t_plus -= lr * grad_V_t_plus
        
        # Compute the gradient w.r.t. V_t_minus
        if V_t_minus is not None:
            grad_V_t_minus = norm_graph * (-param_alpha * S_t * V_t_minus + R_t * V_t_minus) + \
                             lambda_V * V_t_minus + \
                             norm_laplacian * L_t * L_t * V_t_minus + \
                             norm_time * (V_t_minus - V_t)
            V_t_minus -= lr * grad_V_t_minus
        
    return date_ym, V_t, w1, w2, we, theta





#def validate(t):    
def validate(input_pkg):
    t, hetergraph = input_pkg
    L_t, A_ts = hetergraph
    date_ym = list_date_seq[t]
    
    key2int_t = dict_key2int[date_ym]
    int2key_t = dict_int2key[date_ym]
    y_t = groundtruth[date_ym]    
    V_t = embeds[date_ym]
    w1 = classifiers['w1']
    w2 = classifiers['w2']
    we = classifiers['we']
    theta = classifiers['theta']
    
    # Fetch data by month and aggregate values by column
    # view_wiretrans, view_customer = fetch_data_by_month(date_ym, trans_offset=offset)
    # if weight_type.startswith('weight'):
    #    view_wiretrans = group_trans_by_sum_weight(view_wiretrans)
    #    view_wiretrans = weight_normalize(view_wiretrans, weight_type=weight_type)
    # elif weight_type.startswith('count_larger'):
    #    view_wiretrans = group_trans_by_count_larger(view_wiretrans, thres=int(weight_type[12:]))
    # G_t = nx.from_pandas_edgelist(view_wiretrans, 'org_cust', 'bnf_cust', edge_attr='norm_wire_amt')
    
    # Construct graph
    # view_embeds = get_edge_embed(date_ym)
    # G_t = nx.from_pandas_edgelist(view_embeds, 'from_vertex', 'to_vertex', 
    #                              edge_attr=edge_attrs,
    #                              create_using=nx.MultiGraph())
    # G_t = nx.relabel_nodes(G_t, key2int_t)
    # L_t = nx.normalized_laplacian_matrix(G_t, nodelist=list(int2key_t.keys()))
    # A_t = nx.to_scipy_sparse_matrix(G_t, nodelist=int2key_t)
    # A_ts = [nx.to_scipy_sparse_matrix(G_t, weight=attr, nodelist=int2key_t) for attr in edge_attrs]
    A_t = scipy.sparse.csr_matrix(A_ts[0].shape)
    for index_attr in range(we.shape[0]):
        tmpA_t = A_ts[index_attr]
        A_t += sigmoid(we[index_attr,0]) * tmpA_t
    
    if (t + 1) < seq_length:
        date_ym_plus = list_date_seq[t + 1]
        
        key2int_t_plus = dict_key2int[date_ym_plus]
        int2key_t_plus = dict_int2key[date_ym_plus]
        index_t_2_t_plus = [key2int_t_plus[x] if x in key2int_t_plus else -1 for x in int2key_t]
        
        # Get V_t_plus aligned in V_t index order
        V_t_plus_orig = embeds[date_ym_plus]
        V_t_plus_orig = np.append(V_t_plus_orig, np.zeros((1, V_t_plus_orig.shape[1])), axis=0)
        V_t_plus = V_t_plus_orig[index_t_2_t_plus]
        
        # Get y_t_plus aligned in y_t index order
        y_t_plus_orig = groundtruth[date_ym_plus]
        y_t_plus_orig = np.append(y_t_plus_orig, np.zeros((1,1)), axis=0)
        y_t_plus = y_t_plus_orig[index_t_2_t_plus]
    else:
        V_t_plus = None
        y_t_plus = None
        key2int_t_plus = None
        int2key_t_plus = None

    if t > 0:
        date_ym_minus = list_date_seq[t - 1]
        
        key2int_t_minus = dict_key2int[date_ym_minus]
        int2key_t_minus = dict_int2key[date_ym_minus]
        index_t_2_t_minus = [key2int_t_minus[x] if x in key2int_t_minus else -1 for x in int2key_t]
        
        # Get V_t_plus aligned in V_t index order
        V_t_minus_orig = embeds[date_ym_minus]
        V_t_minus_orig = np.append(V_t_minus_orig, np.zeros((1, V_t_minus_orig.shape[1])), axis=0)
        V_t_minus = V_t_minus_orig[index_t_2_t_minus]
                        
        y_t_minus_orig = groundtruth[date_ym_minus]
        y_t_minus_orig = np.append(y_t_minus_orig, np.zeros((1,1)), axis=0)
        y_t_minus = y_t_minus_orig[index_t_2_t_minus] 
    else:
        V_t_minus = None
        y_t_minus = None
        key2int_t_minus = None
        int2key_t_minus = None
    
    # Create sparse matrix S
    vec_i, vec_j, vec_v = scipy.sparse.find(A_t)
    vec_data = np.log(sigmoid(np.sum(V_t[vec_i, :] * V_t[vec_j, :], axis=1)))
    
    # Create sparse matrix R
    l = [[u]*L for u in range(A_t.shape[0])]
    smpl_i = [item for sublist in l for item in sublist]
    l = [list(np.random.choice(A_t.shape[0], min(L*A_t[u,:].nnz, A_t.shape[0]), replace=False)) for u in range(A_t.shape[0])]
    smpl_j = [item for sublist in l for item in sublist]
    smpl_index = list(set(zip(smpl_i, smpl_j)) | set(zip(vec_i, vec_j)))
    smpl_i, smpl_j = zip(*smpl_index)
    smpl_data = np.log(sigmoid(-np.sum(V_t[smpl_i, :] * V_t[smpl_j, :], axis=1)))
    
    # Compute normalization term
    norm_graph = 1 / (vec_data.shape[0] + smpl_data.shape[0])
    norm_laplacian = 1 / (V_t.shape[0] * V_t.shape[1])
    norm_time = 1 / (V_t.shape[0] * V_t.shape[1])
    norm_pred = 10 / V_t.shape[0]
    
    loss_t_graph = norm_graph * (-param_alpha * np.sum(vec_v * vec_data) - np.sum(smpl_data)) + \
                   0.5 * lambda_V * np.sum(V_t ** 2)
    
    loss_t_laplacian = 0.5 * norm_laplacian * np.sum((L_t * V_t) ** 2)
    
    loss_t_time = 0
    if V_t_plus is not None:
        loss_t_time += 0.5 * norm_time * np.sum((V_t - V_t_plus) ** 2)
    if V_t_minus is not None:
        loss_t_time += 0.5 * norm_time * np.sum((V_t - V_t_minus) ** 2)
    
    tmp_recur = np.cos(theta) * V_t + np.sin(theta) * V_t_minus if V_t_minus is not None else V_t
    pred_t = sigmoid(np.dot(tmp_recur, w1) + 0.5 * np.dot(tmp_recur**2, w2))
    vec_one = np.ones((y_t.shape[0], 1))
    q_t = (vec_one - y_t) * (vec_one - pred_t) + y_t * pred_t + 1e-10
    q_t[q_t > 1] = 1.
    loss_t_pred = -norm_pred * np.sum(focal_alpha * (vec_one - q_t)**focal_gamma * np.log(q_t))
    
    loss_t = loss_t_graph + loss_t_laplacian + loss_t_time + loss_t_pred
    
    return loss_t, loss_t_graph, loss_t_laplacian, loss_t_time, loss_t_pred





dict_int2key = {}
dict_key2int = {}
for date_ym in list_date_seq:
    with open("GraphLaplacian/offset_3/aggregate/weight_none_nodes_int2key_{}-{}.pickle".format(
        date_ym[0], date_ym[1]), 'rb') as f:
        int2key_t = pickle.load(f)
        dict_int2key[date_ym] = int2key_t
    with open("GraphLaplacian/offset_3/aggregate/weight_none_nodes_key2int_{}-{}.pickle".format(
        date_ym[0], date_ym[1]), 'rb') as f:
        key2int_t = pickle.load(f)
        dict_key2int[date_ym] = key2int_t




int2key_t = dict_int2key[(2018, 1)]
key2int_t_plus = dict_key2int[(2018, 2)]
int2key_t_plus = dict_int2key[(2018, 2)]
print(int2key_t[0])
print(int2key_t_plus[0])



groundtruth = {}
for date_ym in list_date_seq:
    year = date_ym[0]
    month = date_ym[1]
    from_date = pd.to_datetime("{}/{}/{}".format(month, 1, year))
    to_date = from_date + pd.DateOffset(months=1)
    
    target_list = SARCase[(SARCase.Status_SAR == 4) & 
                          (SARCase.created_date > from_date) & 
                          (SARCase.created_date < to_date)]['customerno'].unique()
    
    dict_int2key_t = dict_int2key[date_ym]
    dict_key2int_t = dict_key2int[date_ym]
    pos_indices = [dict_key2int_t[x] for x in target_list if x in dict_key2int_t]
    
    Y_t = np.zeros((len(dict_int2key[date_ym]), 1))
    Y_t[pos_indices, 0] = 1
    groundtruth[date_ym] = Y_t
    print("Number of instances with positive label in {}: {}".format(date_ym, int(np.sum(Y_t))))


'''
Number of instances with positive label in (2018, 1): 13
Number of instances with positive label in (2018, 2): 14
Number of instances with positive label in (2018, 3): 11
Number of instances with positive label in (2018, 4): 16
Number of instances with positive label in (2018, 5): 6
Number of instances with positive label in (2018, 6): 9
Number of instances with positive label in (2018, 7): 14
Number of instances with positive label in (2018, 8): 37
Number of instances with positive label in (2018, 9): 12
Number of instances with positive label in (2018, 10): 9
Number of instances with positive label in (2018, 11): 21
Number of instances with positive label in (2018, 12): 19
Number of instances with positive label in (2019, 1): 8
Number of instances with positive label in (2019, 2): 4
Number of instances with positive label in (2019, 3): 2
Number of instances with positive label in (2019, 4): 5
Number of instances with positive label in (2019, 5): 4
Number of instances with positive label in (2019, 6): 4
'''



dict_hetergraph = {}
for t in range(seq_length):
    date_ym = list_date_seq[t]
    key2int_t = dict_key2int[date_ym]
    int2key_t = dict_int2key[date_ym]
    print('Processing heterogeneous graph at {} ......'.format(date_ym))
    
    view_embeds = get_edge_embed(date_ym)
    G_t = nx.from_pandas_edgelist(view_embeds, 'from_vertex', 'to_vertex', 
                                  edge_attr=edge_attrs,
                                  create_using=nx.MultiGraph())
    G_t = nx.relabel_nodes(G_t, key2int_t)
    L_t = nx.normalized_laplacian_matrix(G_t, nodelist=list(int2key_t.keys()))
    A_t = nx.to_scipy_sparse_matrix(G_t, nodelist=int2key_t)
    A_ts = [nx.to_scipy_sparse_matrix(G_t, weight=attr, nodelist=int2key_t) for attr in edge_attrs]
    
    dict_hetergraph[date_ym] = (L_t, A_ts)
    print('Finished L_t.shape = {}'.format(A_t.shape, L_t.shape))



'''
Processing heterogeneous graph at (2018, 1) ......
Finished L_t.shape = (66627, 66627)
Processing heterogeneous graph at (2018, 2) ......
Finished L_t.shape = (64701, 64701)
Processing heterogeneous graph at (2018, 3) ......
Finished L_t.shape = (64712, 64712)
Processing heterogeneous graph at (2018, 4) ......
Finished L_t.shape = (65481, 65481)
Processing heterogeneous graph at (2018, 5) ......
Finished L_t.shape = (64407, 64407)
Processing heterogeneous graph at (2018, 6) ......
Finished L_t.shape = (66627, 66627)
Processing heterogeneous graph at (2018, 7) ......
Finished L_t.shape = (67397, 67397)
Processing heterogeneous graph at (2018, 8) ......
Finished L_t.shape = (66943, 66943)
Processing heterogeneous graph at (2018, 9) ......
Finished L_t.shape = (65069, 65069)
Processing heterogeneous graph at (2018, 10) ......
Finished L_t.shape = (65047, 65047)
Processing heterogeneous graph at (2018, 11) ......
Finished L_t.shape = (63888, 63888)
Processing heterogeneous graph at (2018, 12) ......
Finished L_t.shape = (63092, 63092)
Processing heterogeneous graph at (2019, 1) ......
Finished L_t.shape = (64078, 64078)
Processing heterogeneous graph at (2019, 2) ......
Finished L_t.shape = (59535, 59535)
Processing heterogeneous graph at (2019, 3) ......
Finished L_t.shape = (58637, 58637)
Processing heterogeneous graph at (2019, 4) ......
Finished L_t.shape = (57278, 57278)
Processing heterogeneous graph at (2019, 5) ......
Finished L_t.shape = (53995, 53995)
Processing heterogeneous graph at (2019, 6) ......
Finished L_t.shape = (55511, 55511)
'''



#
# Variables Initialization
# ---------------------------------------------------------------------------------------------------------------
embeds = {}
for date_ym in list_date_seq:
    embeds[date_ym] = np.random.rand(len(dict_key2int[date_ym]), latent_dim)

classifiers = {'w1': np.random.rand(latent_dim, 1),
               'w2': np.random.rand(latent_dim, 1),
               'we': np.random.rand(len(edge_attrs), 1),
               'theta': np.random.rand()}


#
# Model Training
# ---------------------------------------------------------------------------------------------------------------
seq_length = len(list_date_seq)
list_input_pkg = []
for t in range(seq_length):
    list_input_pkg.append((t, dict_hetergraph[list_date_seq[t]]))
num_cores = 9
print("Training ......")
for epoch in range(epochs):        
    #                              #
    # Embedding training           #
    # ---------------------------- #
    res_list = []
    with Pool(processes=num_cores) as p:
        max_ = seq_length
        with tqdm_notebook(total=max_) as pbar:
            #for t, res in tqdm_notebook(enumerate(p.imap_unordered(update_params, range(seq_length)))):
            for input_pkg, res in tqdm_notebook(enumerate(p.imap_unordered(update_params, list_input_pkg))):
                pbar.update()
                res_list.append(res)
    
    date_ym_seq, V_t_seq, w1_seq, w2_seq, we_seq, theta_seq = zip(*res_list)
    for index in range(len(date_ym_seq)):
        date_ym = date_ym_seq[index]
        embeds[date_ym] = V_t_seq[index]
    
    classifiers['w1'] = np.mean(w1_seq, axis=0)
    classifiers['w2'] = np.mean(w2_seq, axis=0)
    tmp_mul = np.exp(np.mean(we_seq, axis=0))
    classifiers['we'] = tmp_mul / np.sum(tmp_mul)
    classifiers['theta'] = np.mean(theta_seq, axis=0)
    
    #                              #
    # Validation                   #
    # ---------------------------- #
    res_list = []
    with Pool(processes=num_cores) as p:
        max_ = seq_length
        with tqdm_notebook(total=max_) as pbar:
            #for t, res in tqdm_notebook(enumerate(p.imap_unordered(validate, range(seq_length)))):
            for input_pkg, res in tqdm_notebook(enumerate(p.imap_unordered(validate, list_input_pkg))):
                pbar.update()
                res_list.append(res)
    
    # Save the checkpoint
    for t in range(seq_length):
        date_ym = list_date_seq[t]
        np.save('graph_embed/offset_{}/aggregate/{}_heter_superv_recur_focal_logisticMF_embed2_{}-{}'.format(
            offset, weight_type, date_ym[0], date_ym[1]), embeds[date_ym])  
    
    vec_loss = np.mean(res_list, axis=0)
    degree = (classifiers['theta'] % (2*np.pi)) / (2*np.pi) * 360
    print("Epoch {} Loss => total: {:.4f} , g: {:.4f}, l: {:.4f}, t: {:.4f}, p: {:.4f}, theta: {:.3f}, theta_d: {:.3f}".
          format(epoch, vec_loss[0], vec_loss[1], vec_loss[2], vec_loss[3], vec_loss[4], classifiers['theta'], degree))





'''
Training ......
100% 18/18 [07:09<00:00, 6.49s/it]
18/|/| 18/? [07:09<00:00, 6.49s/it]
100% 18/18 [05:37<00:00, 4.68s/it]
18/|/| 18/? [05:37<00:00, 4.68s/it]
Epoch 0 Loss => total: 59.1698 , g: 35.7363, l: 0.1009, t: 0.3109, p: 23.0217, theta: 0.517, theta_d: 29.644
100% 18/18 [07:14<00:00, 7.66s/it]
18/|/| 18/? [07:14<00:00, 7.66s/it]
100% 18/18 [06:03<00:00, 5.84s/it]
18/|/| 18/? [06:03<00:00, 5.84s/it]
Epoch 1 Loss => total: 58.7724 , g: 35.3438, l: 0.1000, t: 0.3069, p: 23.0217, theta: 0.517, theta_d: 29.644
100% 18/18 [07:02<00:00, 7.62s/it]
18/|/| 18/? [07:02<00:00, 7.62s/it]
100% 18/18 [05:48<00:00, 5.82s/it]
18/|/| 18/? [05:48<00:00, 5.82s/it]
Epoch 2 Loss => total: 58.3808 , g: 34.9569, l: 0.0992, t: 0.3030, p: 23.0217, theta: 0.517, theta_d: 29.644
100% 18/18 [07:20<00:00, 7.42s/it]
18/|/| 18/? [07:20<00:00, 7.42s/it]
100% 18/18 [05:48<00:00, 5.29s/it]
18/|/| 18/? [05:48<00:00, 5.29s/it]
Epoch 3 Loss => total: 57.9948 , g: 34.5756, l: 0.0983, t: 0.2992, p: 23.0216, theta: 0.517, theta_d: 29.644
100% 18/18 [07:13<00:00, 7.22s/it]
18/|/| 18/? [07:13<00:00, 7.22s/it]
100% 18/18 [05:58<00:00, 5.86s/it]
18/|/| 18/? [05:58<00:00, 5.86s/it]
Epoch 4 Loss => total: 57.6087 , g: 34.1944, l: 0.0975, t: 0.2955, p: 23.0214, theta: 0.517, theta_d: 29.635
100% 18/18 [07:21<00:00, 7.64s/it]
18/|/| 18/? [07:21<00:00, 7.64s/it]
100% 18/18 [05:54<00:00, 6.26s/it]
18/|/| 18/? [05:54<00:00, 6.26s/it]
Epoch 5 Loss => total: 57.2321 , g: 33.8225, l: 0.0966, t: 0.2918, p: 23.0212, theta: 0.517, theta_d: 29.634
100% 18/18 [07:21<00:00, 7.31s/it]
18/|/| 18/? [07:21<00:00, 7.31s/it]
100% 18/18 [06:09<00:00, 5.83s/it]
18/|/| 18/? [06:09<00:00, 5.83s/it]
Epoch 6 Loss => total: 56.8603 , g: 33.4554, l: 0.0958, t: 0.2881, p: 23.0210, theta: 0.517, theta_d: 29.624
100% 18/18 [07:26<00:00, 7.02s/it]
18/|/| 18/? [07:26<00:00, 7.01s/it]
100% 18/18 [05:59<00:00, 5.79s/it]
18/|/| 18/? [05:59<00:00, 5.79s/it]
Epoch 7 Loss => total: 56.4926 , g: 33.0923, l: 0.0950, t: 0.2845, p: 23.0208, theta: 0.517, theta_d: 29.618
100% 18/18 [07:31<00:00, 8.21s/it]
18/|/| 18/? [07:31<00:00, 8.21s/it]
100% 18/18 [05:51<00:00, 7.42s/it]
18/|/| 18/? [05:51<00:00, 7.42s/it]
Epoch 8 Loss => total: 56.1313 , g: 32.7356, l: 0.0941, t: 0.2809, p: 23.0206, theta: 0.517, theta_d: 29.617
100% 18/18 [07:18<00:00, 7.33s/it]
18/|/| 18/? [07:18<00:00, 7.33s/it]
100% 18/18 [06:01<00:00, 6.45s/it]
18/|/| 18/? [06:01<00:00, 6.45s/it]
Epoch 9 Loss => total: 55.7732 , g: 32.3822, l: 0.0933, t: 0.2774, p: 23.0204, theta: 0.517, theta_d: 29.600
100% 18/18 [07:40<00:00, 7.41s/it]
18/|/| 18/? [07:40<00:00, 7.41s/it]
100% 18/18 [05:49<00:00, 5.79s/it]
18/|/| 18/? [05:49<00:00, 5.79s/it]
Epoch 10 Loss => total: 55.4125 , g: 32.0263, l: 0.0924, t: 0.2738, p: 23.0199, theta: 0.516, theta_d: 29.547
100% 18/18 [06:43<00:00, 6.77s/it]
18/|/| 18/? [06:43<00:00, 6.77s/it]
100% 18/18 [05:41<00:00, 7.46s/it]
18/|/| 18/? [05:41<00:00, 7.46s/it]
Epoch 11 Loss => total: 55.0361 , g: 31.6554, l: 0.0915, t: 0.2701, p: 23.0190, theta: 0.514, theta_d: 29.466
100% 18/18 [07:01<00:00, 8.62s/it]
18/|/| 18/? [07:01<00:00, 8.62s/it]
100% 18/18 [05:43<00:00, 5.34s/it]
18/|/| 18/? [05:43<00:00, 5.34s/it]
Epoch 12 Loss => total: 54.6389 , g: 31.2645, l: 0.0906, t: 0.2662, p: 23.0176, theta: 0.512, theta_d: 29.320
100% 18/18 [07:34<00:00, 8.46s/it]
18/|/| 18/? [07:34<00:00, 8.46s/it]
100% 18/18 [05:54<00:00, 6.59s/it]
18/|/| 18/? [05:54<00:00, 6.59s/it]
Epoch 13 Loss => total: 31.2334 , g: 30.8792, l: 0.0896, t: 0.2624, p: 0.0022, theta: 0.400, theta_d: 22.933
100% 18/18 [07:32<00:00, 7.41s/it]
18/|/| 18/? [07:32<00:00, 7.41s/it]
100% 18/18 [05:42<00:00, 5.49s/it]
18/|/| 18/? [05:42<00:00, 5.49s/it]
Epoch 14 Loss => total: 30.9328 , g: 30.5830, l: 0.0889, t: 0.2594, p: 0.0015, theta: 7.722, theta_d: 82.423
100% 18/18 [07:15<00:00, 5.29s/it]
18/|/| 18/? [07:15<00:00, 5.29s/it]
100% 18/18 [05:55<00:00, 5.85s/it]
18/|/| 18/? [05:55<00:00, 5.85s/it]
Epoch 15 Loss => total: 52.3731 , g: 30.2876, l: 0.0882, t: 0.2565, p: 21.7409, theta: 14.877, theta_d: 132.415
100% 18/18 [07:01<00:00, 6.95s/it]
18/|/| 18/? [07:01<00:00, 6.95s/it]
100% 18/18 [05:37<00:00, 8.04s/it]
18/|/| 18/? [05:37<00:00, 8.04s/it]
Epoch 16 Loss => total: 52.0664 , g: 29.9850, l: 0.0874, t: 0.2535, p: 21.7405, theta: 14.384, theta_d: 104.151
100% 18/18 [06:46<00:00, 8.60s/it]
18/|/| 18/? [06:46<00:00, 8.60s/it]
100% 18/18 [05:27<00:00, 5.30s/it]
18/|/| 18/? [05:27<00:00, 5.30s/it]
Epoch 17 Loss => total: 51.7698 , g: 29.6923, l: 0.0867, t: 0.2506, p: 21.7402, theta: 22.100, theta_d: 186.257
100% 18/18 [06:39<00:00, 7.78s/it]
18/|/| 18/? [06:39<00:00, 7.78s/it]
100% 18/18 [05:28<00:00, 5.75s/it]
18/|/| 18/? [05:28<00:00, 5.75s/it]
Epoch 18 Loss => total: 29.7182 , g: 29.3816, l: 0.0859, t: 0.2475, p: 0.0031, theta: 24.298, theta_d: 312.176
100% 18/18 [07:07<00:00, 7.12s/it]
18/|/| 18/? [07:07<00:00, 7.12s/it]
100% 18/18 [05:32<00:00, 5.49s/it]
18/|/| 18/? [05:32<00:00, 5.49s/it]
Epoch 19 Loss => total: 29.4209 , g: 29.0891, l: 0.0852, t: 0.2446, p: 0.0020, theta: 25.047, theta_d: 355.106
100% 18/18 [06:37<00:00, 8.11s/it]
18/|/| 18/? [06:37<00:00, 8.11s/it]
100% 18/18 [05:27<00:00, 5.28s/it]
18/|/| 18/? [05:27<00:00, 5.28s/it]
Epoch 20 Loss => total: 29.1379 , g: 28.8094, l: 0.0845, t: 0.2418, p: 0.0022, theta: 26.017, theta_d: 50.687
100% 18/18 [06:31<00:00, 10.58s/it]
18/|/| 18/? [06:31<00:00, 10.58s/it]
100% 18/18 [05:20<00:00, 5.69s/it]
18/|/| 18/? [05:20<00:00, 5.69s/it]
Epoch 21 Loss => total: 28.8698 , g: 28.5448, l: 0.0838, t: 0.2392, p: 0.0020, theta: 24.870, theta_d: 344.933
100% 18/18 [06:08<00:00, 6.42s/it]
18/|/| 18/? [06:08<00:00, 6.42s/it]
100% 18/18 [05:18<00:00, 5.45s/it]
18/|/| 18/? [05:18<00:00, 5.45s/it]
Epoch 22 Loss => total: 50.3297 , g: 28.2730, l: 0.0831, t: 0.2365, p: 21.7370, theta: 27.619, theta_d: 142.470
100% 18/18 [06:17<00:00, 9.23s/it]
18/|/| 18/? [06:17<00:00, 9.23s/it]
100% 18/18 [05:08<00:00, 5.63s/it]
18/|/| 18/? [05:08<00:00, 5.63s/it]
Epoch 23 Loss => total: 50.0502 , g: 27.9973, l: 0.0824, t: 0.2338, p: 21.7367, theta: 27.493, theta_d: 135.247
100% 18/18 [06:09<00:00, 7.01s/it]
18/|/| 18/? [06:09<00:00, 7.01s/it]
100% 18/18 [05:11<00:00, 4.78s/it]
18/|/| 18/? [05:11<00:00, 4.78s/it]
Epoch 24 Loss => total: 28.0418 , g: 27.7267, l: 0.0817, t: 0.2311, p: 0.0023, theta: 24.576, theta_d: 328.085
100% 18/18 [06:09<00:00, 6.15s/it]
18/|/| 18/? [06:09<00:00, 6.15s/it]
100% 18/18 [05:06<00:00, 4.96s/it]
18/|/| 18/? [05:06<00:00, 4.96s/it]
Epoch 25 Loss => total: 49.5097 , g: 27.4645, l: 0.0810, t: 0.2285, p: 21.7357, theta: 27.564, theta_d: 139.288
100% 18/18 [06:05<00:00, 7.31s/it]
18/|/| 18/? [06:05<00:00, 7.31s/it]
100% 18/18 [05:06<00:00, 4.87s/it]
18/|/| 18/? [05:06<00:00, 4.87s/it]
Epoch 26 Loss => total: 49.2427 , g: 27.2012, l: 0.0803, t: 0.2259, p: 21.7352, theta: 33.799, theta_d: 136.536
100% 18/18 [06:21<00:00, 6.91s/it]
18/|/| 18/? [06:21<00:00, 6.91s/it]
100% 18/18 [05:25<00:00, 6.31s/it]
18/|/| 18/? [05:25<00:00, 6.31s/it]
Epoch 27 Loss => total: 27.2471 , g: 26.9416, l: 0.0796, t: 0.2233, p: 0.0026, theta: 45.131, theta_d: 65.792
100% 18/18 [06:13<00:00, 5.82s/it]
18/|/| 18/? [06:13<00:00, 5.82s/it]
100% 18/18 [05:05<00:00, 4.88s/it]
18/|/| 18/? [05:05<00:00, 4.88s/it]
Epoch 28 Loss => total: 48.7331 , g: 26.6989, l: 0.0789, t: 0.2209, p: 21.7344, theta: 47.188, theta_d: 183.668
100% 18/18 [06:11<00:00, 6.64s/it]
18/|/| 18/? [06:11<00:00, 6.64s/it]
100% 18/18 [05:04<00:00, 5.11s/it]
18/|/| 18/? [05:04<00:00, 5.11s/it]
Epoch 29 Loss => total: 48.4578 , g: 26.4276, l: 0.0782, t: 0.2183, p: 21.7337, theta: 40.573, theta_d: 164.667
100% 18/18 [06:02<00:00, 7.84s/it]
18/|/| 18/? [06:02<00:00, 7.84s/it]
100% 18/18 [05:08<00:00, 6.59s/it]
18/|/| 18/? [05:08<00:00, 6.59s/it]
Epoch 30 Loss => total: 26.4650 , g: 26.1692, l: 0.0775, t: 0.2157, p: 0.0026, theta: 48.865, theta_d: 279.771
100% 18/18 [06:00<00:00, 6.72s/it]
18/|/| 18/? [06:00<00:00, 6.72s/it]
100% 18/18 [05:06<00:00, 5.89s/it]
18/|/| 18/? [05:06<00:00, 5.89s/it]
Epoch 31 Loss => total: 47.9337 , g: 25.9113, l: 0.0768, t: 0.2131, p: 21.7325, theta: 46.631, theta_d: 151.750
100% 18/18 [06:01<00:00, 5.34s/it]
18/|/| 18/? [06:01<00:00, 5.34s/it]
100% 18/18 [05:03<00:00, 4.88s/it]
18/|/| 18/? [05:03<00:00, 4.88s/it]
Epoch 32 Loss => total: 47.6816 , g: 25.6632, l: 0.0761, t: 0.2107, p: 21.7317, theta: 47.979, theta_d: 229.001
100% 18/18 [06:02<00:00, 6.66s/it]
18/|/| 18/? [06:02<00:00, 6.66s/it]
100% 18/18 [05:04<00:00, 16.90s/it]
18/|/| 18/? [05:04<00:00, 16.90s/it]
Epoch 33 Loss => total: 25.6944 , g: 25.4063, l: 0.0754, t: 0.2081, p: 0.0046, theta: 48.985, theta_d: 286.610
100% 18/18 [06:01<00:00, 7.69s/it]
18/|/| 18/? [06:01<00:00, 7.69s/it]
100% 18/18 [05:02<00:00, 5.06s/it]
18/|/| 18/? [05:02<00:00, 5.06s/it]
Epoch 34 Loss => total: 47.1740 , g: 25.1627, l: 0.0747, t: 0.2057, p: 21.7310, theta: 47.317, theta_d: 191.075
100% 18/18 [06:01<00:00, 5.95s/it]
18/|/| 18/? [06:01<00:00, 5.95s/it]
100% 18/18 [05:03<00:00, 5.04s/it]
18/|/| 18/? [05:03<00:00, 5.04s/it]
Epoch 35 Loss => total: 46.9204 , g: 24.9133, l: 0.0740, t: 0.2032, p: 21.7299, theta: 47.035, theta_d: 174.894
100% 18/18 [06:00<00:00, 6.08s/it]
18/|/| 18/? [06:00<00:00, 6.08s/it]
100% 18/18 [05:03<00:00, 4.64s/it]
18/|/| 18/? [05:03<00:00, 4.64s/it]
Epoch 36 Loss => total: 46.6736 , g: 24.6704, l: 0.0733, t: 0.2008, p: 21.7290, theta: 48.065, theta_d: 233.935
100% 18/18 [06:01<00:00, 6.55s/it]
18/|/| 18/? [06:01<00:00, 6.55s/it]
100% 18/18 [05:04<00:00, 5.56s/it]
18/|/| 18/? [05:04<00:00, 5.56s/it]
Epoch 37 Loss => total: 46.4271 , g: 24.4280, l: 0.0726, t: 0.1984, p: 21.7280, theta: 42.082, theta_d: 251.103
100% 18/18 [06:00<00:00, 5.88s/it]
18/|/| 18/? [06:00<00:00, 5.88s/it]
100% 18/18 [05:03<00:00, 6.26s/it]
18/|/| 18/? [05:02<00:00, 6.25s/it]
Epoch 38 Loss => total: 24.4681 , g: 24.1899, l: 0.0720, t: 0.1961, p: 0.0102, theta: 44.927, theta_d: 54.114
100% 18/18 [05:59<00:00, 5.70s/it]
18/|/| 18/? [05:59<00:00, 5.70s/it]
100% 18/18 [05:05<00:00, 5.29s/it]
18/|/| 18/? [05:05<00:00, 5.29s/it]
Epoch 39 Loss => total: 24.2501 , g: 23.9820, l: 0.0713, t: 0.1939, p: 0.0029, theta: 44.340, theta_d: 20.509
100% 18/18 [06:03<00:00, 6.71s/it]
18/|/| 18/? [06:03<00:00, 6.71s/it]
100% 18/18 [05:02<00:00, 4.70s/it]
18/|/| 18/? [05:02<00:00, 4.70s/it]
Epoch 40 Loss => total: 24.0337 , g: 23.7686, l: 0.0707, t: 0.1918, p: 0.0026, theta: 43.249, theta_d: 318.004
100% 18/18 [05:59<00:00, 6.00s/it]
18/|/| 18/? [05:59<00:00, 6.00s/it]
100% 18/18 [05:02<00:00, 6.29s/it]
18/|/| 18/? [05:02<00:00, 6.29s/it]
Epoch 41 Loss => total: 45.5328 , g: 23.5512, l: 0.0701, t: 0.1897, p: 21.7219, theta: 39.349, theta_d: 94.558
100% 18/18 [06:04<00:00, 6.12s/it]
18/|/| 18/? [06:04<00:00, 6.12s/it]
100% 18/18 [05:02<00:00, 4.78s/it]
18/|/| 18/? [05:02<00:00, 4.78s/it]
Epoch 42 Loss => total: 45.3260 , g: 23.3433, l: 0.0695, t: 0.1876, p: 21.7256, theta: 46.015, theta_d: 116.488
100% 18/18 [06:02<00:00, 5.71s/it]
18/|/| 18/? [06:02<00:00, 5.71s/it]
100% 18/18 [05:02<00:00, 5.56s/it]
18/|/| 18/? [05:02<00:00, 5.56s/it]
Epoch 43 Loss => total: 23.3925 , g: 23.1341, l: 0.0688, t: 0.1856, p: 0.0040, theta: 49.197, theta_d: 298.758
100% 18/18 [06:03<00:00, 6.01s/it]
18/|/| 18/? [06:03<00:00, 6.01s/it]
100% 18/18 [05:04<00:00, 5.27s/it]
18/|/| 18/? [05:04<00:00, 5.27s/it]
Epoch 44 Loss => total: 44.8960 , g: 22.9218, l: 0.0682, t: 0.1834, p: 21.7226, theta: 45.707, theta_d: 98.843
100% 18/18 [05:58<00:00, 19.92s/it]
18/|/| 18/? [05:58<00:00, 19.92s/it]
100% 18/18 [05:03<00:00, 5.83s/it]
18/|/| 18/? [05:03<00:00, 5.83s/it]
Epoch 45 Loss => total: 22.9721 , g: 22.7198, l: 0.0676, t: 0.1814, p: 0.0032, theta: 50.429, theta_d: 9.383
100% 18/18 [06:00<00:00, 6.06s/it]
18/|/| 18/? [06:00<00:00, 6.06s/it]
100% 18/18 [05:09<00:00, 6.17s/it]
18/|/| 18/? [05:09<00:00, 6.17s/it]
Epoch 46 Loss => total: 22.7719 , g: 22.5234, l: 0.0670, t: 0.1795, p: 0.0019, theta: 51.681, theta_d: 81.125
100% 18/18 [06:01<00:00, 6.08s/it]
18/|/| 18/? [06:01<00:00, 6.08s/it]
100% 18/18 [05:06<00:00, 5.91s/it]
18/|/| 18/? [05:06<00:00, 5.91s/it]
Epoch 47 Loss => total: 44.2969 , g: 22.3302, l: 0.0665, t: 0.1776, p: 21.7227, theta: 54.406, theta_d: 237.256
100% 18/18 [06:02<00:00, 6.05s/it]
18/|/| 18/? [06:02<00:00, 6.05s/it]
100% 18/18 [05:04<00:00, 5.43s/it]
18/|/| 18/? [05:04<00:00, 5.43s/it]
Epoch 48 Loss => total: 22.3645 , g: 22.1178, l: 0.0659, t: 0.1755, p: 0.0053, theta: 57.834, theta_d: 73.638
100% 18/18 [06:07<00:00, 7.32s/it]
18/|/| 18/? [06:07<00:00, 7.32s/it]
100% 18/18 [05:01<00:00, 16.77s/it]
18/|/| 18/? [05:01<00:00, 16.76s/it]
Epoch 49 Loss => total: 43.8953 , g: 21.9349, l: 0.0652, t: 0.1736, p: 21.7215, theta: 60.722, theta_d: 239.122
100% 18/18 [05:59<00:00, 5.51s/it]
18/|/| 18/? [05:59<00:00, 5.51s/it]
100% 18/18 [05:03<00:00, 5.20s/it]
18/|/| 18/? [05:03<00:00, 5.20s/it]
Epoch 50 Loss => total: 21.9677 , g: 21.7249, l: 0.0647, t: 0.1716, p: 0.0066, theta: 56.398, theta_d: 351.376
100% 18/18 [06:01<00:00, 6.50s/it]
18/|/| 18/? [06:01<00:00, 6.50s/it]
100% 18/18 [05:02<00:00, 5.03s/it]
18/|/| 18/? [05:02<00:00, 5.03s/it]
Epoch 51 Loss => total: 21.7763 , g: 21.5392, l: 0.0641, t: 0.1698, p: 0.0031, theta: 55.146, theta_d: 279.638
100% 18/18 [06:03<00:00, 6.26s/it]
18/|/| 18/? [06:03<00:00, 6.26s/it]
100% 18/18 [05:07<00:00, 6.15s/it]
18/|/| 18/? [05:07<00:00, 6.15s/it]
Epoch 52 Loss => total: 43.2939 , g: 21.3438, l: 0.0635, t: 0.1678, p: 21.7188, theta: 53.578, theta_d: 189.781
100% 18/18 [05:58<00:00, 5.52s/it]
18/|/| 18/? [05:58<00:00, 5.52s/it]
100% 18/18 [05:04<00:00, 5.98s/it]
18/|/| 18/? [05:04<00:00, 5.98s/it]
Epoch 53 Loss => total: 43.0902 , g: 21.1446, l: 0.0629, t: 0.1659, p: 21.7167, theta: 53.769, theta_d: 200.743
100% 18/18 [06:04<00:00, 6.03s/it]
18/|/| 18/? [06:04<00:00, 6.03s/it]
100% 18/18 [05:04<00:00, 4.96s/it]
18/|/| 18/? [05:04<00:00, 4.96s/it]
Epoch 54 Loss => total: 21.1864 , g: 20.9483, l: 0.0623, t: 0.1639, p: 0.0119, theta: 51.330, theta_d: 60.986
100% 18/18 [06:05<00:00, 7.10s/it]
18/|/| 18/? [06:05<00:00, 7.10s/it]
100% 18/18 [05:03<00:00, 5.17s/it]
18/|/| 18/? [05:03<00:00, 5.17s/it]
Epoch 55 Loss => total: 21.0106 , g: 20.7833, l: 0.0617, t: 0.1622, p: 0.0033, theta: 55.528, theta_d: 301.537
100% 18/18 [06:02<00:00, 6.22s/it]
18/|/| 18/? [06:02<00:00, 6.22s/it]
100% 18/18 [05:02<00:00, 16.79s/it]
18/|/| 18/? [05:02<00:00, 16.78s/it]
Epoch 56 Loss => total: 20.8228 , g: 20.5994, l: 0.0612, t: 0.1604, p: 0.0018, theta: 58.037, theta_d: 85.278
100% 18/18 [06:00<00:00, 5.75s/it]
18/|/| 18/? [06:00<00:00, 5.75s/it]
100% 18/18 [05:01<00:00, 4.84s/it]
18/|/| 18/? [05:01<00:00, 4.84s/it]
Epoch 57 Loss => total: 42.3607 , g: 20.4274, l: 0.0607, t: 0.1587, p: 21.7139, theta: 60.465, theta_d: 224.413
100% 18/18 [06:01<00:00, 6.33s/it]
18/|/| 18/? [06:01<00:00, 6.33s/it]
100% 18/18 [05:03<00:00, 5.03s/it]
18/|/| 18/? [05:03<00:00, 5.03s/it]
Epoch 58 Loss => total: 20.4659 , g: 20.2407, l: 0.0601, t: 0.1569, p: 0.0083, theta: 57.254, theta_d: 40.422
100% 18/18 [05:58<00:00, 19.89s/it]
18/|/| 18/? [05:58<00:00, 19.89s/it]
100% 18/18 [05:05<00:00, 5.77s/it]
18/|/| 18/? [05:05<00:00, 5.77s/it]
Epoch 59 Loss => total: 42.0174 , g: 20.0857, l: 0.0595, t: 0.1553, p: 21.7169, theta: 59.424, theta_d: 164.719
100% 18/18 [05:59<00:00, 5.49s/it]
18/|/| 18/? [05:59<00:00, 5.49s/it]
100% 18/18 [05:05<00:00, 4.70s/it]
18/|/| 18/? [05:05<00:00, 4.70s/it]
Epoch 60 Loss => total: 20.1221 , g: 19.9056, l: 0.0590, t: 0.1535, p: 0.0040, theta: 63.909, theta_d: 61.719
100% 18/18 [06:00<00:00, 6.09s/it]
18/|/| 18/? [06:00<00:00, 6.09s/it]
100% 18/18 [05:03<00:00, 5.25s/it]
18/|/| 18/? [05:03<00:00, 5.25s/it]
Epoch 61 Loss => total: 19.9591 , g: 19.7455, l: 0.0585, t: 0.1519, p: 0.0031, theta: 61.964, theta_d: 310.269
100% 18/18 [06:02<00:00, 6.15s/it]
18/|/| 18/? [06:02<00:00, 6.15s/it]
100% 18/18 [05:01<00:00, 5.63s/it]
18/|/| 18/? [05:01<00:00, 5.63s/it]
Epoch 62 Loss => total: 19.7880 , g: 19.5776, l: 0.0580, t: 0.1503, p: 0.0021, theta: 57.995, theta_d: 82.845
100% 18/18 [05:58<00:00, 5.71s/it]
18/|/| 18/? [05:58<00:00, 5.71s/it]
100% 18/18 [05:02<00:00, 5.01s/it]
18/|/| 18/? [05:02<00:00, 5.01s/it]
Epoch 63 Loss => total: 41.3372 , g: 19.4204, l: 0.0575, t: 0.1487, p: 21.7106, theta: 60.689, theta_d: 237.251
100% 18/18 [06:04<00:00, 7.39s/it]
18/|/| 18/? [06:04<00:00, 7.39s/it]
100% 18/18 [05:02<00:00, 4.61s/it]
18/|/| 18/? [05:02<00:00, 4.61s/it]
Epoch 64 Loss => total: 41.1590 , g: 19.2463, l: 0.0570, t: 0.1470, p: 21.7087, theta: 60.262, theta_d: 212.783
100% 18/18 [06:04<00:00, 7.39s/it]
18/|/| 18/? [06:04<00:00, 7.39s/it]
100% 18/18 [05:04<00:00, 5.09s/it]
18/|/| 18/? [05:04<00:00, 5.09s/it]
Epoch 65 Loss => total: 40.9713 , g: 19.0759, l: 0.0565, t: 0.1453, p: 21.6935, theta: 61.175, theta_d: 265.044
100% 18/18 [06:01<00:00, 6.06s/it]
18/|/| 18/? [06:01<00:00, 6.06s/it]
100% 18/18 [05:02<00:00, 16.78s/it]
18/|/| 18/? [05:02<00:00, 16.78s/it]
Epoch 66 Loss => total: 19.1236 , g: 18.9071, l: 0.0560, t: 0.1437, p: 0.0168, theta: 57.470, theta_d: 52.810
100% 18/18 [06:03<00:00, 6.54s/it]
18/|/| 18/? [06:03<00:00, 6.54s/it]
100% 18/18 [05:02<00:00, 5.60s/it]
18/|/| 18/? [05:02<00:00, 5.60s/it]
Epoch 67 Loss => total: 40.6738 , g: 18.7677, l: 0.0554, t: 0.1422, p: 21.7085, theta: 59.599, theta_d: 174.785
100% 18/18 [06:00<00:00, 6.22s/it]
18/|/| 18/? [06:00<00:00, 6.22s/it]
100% 18/18 [05:04<00:00, 5.12s/it]
18/|/| 18/? [05:04<00:00, 5.12s/it]
Epoch 68 Loss => total: 18.8035 , g: 18.6025, l: 0.0549, t: 0.1406, p: 0.0055, theta: 61.930, theta_d: 308.309
100% 18/18 [06:01<00:00, 5.84s/it]
18/|/| 18/? [06:01<00:00, 5.83s/it]
100% 18/18 [05:02<00:00, 4.88s/it]
18/|/| 18/? [05:02<00:00, 4.88s/it]
Epoch 69 Loss => total: 40.3468 , g: 18.4484, l: 0.0544, t: 0.1391, p: 21.7050, theta: 60.319, theta_d: 216.001
100% 18/18 [05:59<00:00, 5.55s/it]
18/|/| 18/? [05:59<00:00, 5.55s/it]
100% 18/18 [05:02<00:00, 4.90s/it]
18/|/| 18/? [05:02<00:00, 4.90s/it]
Epoch 70 Loss => total: 40.1816 , g: 18.2872, l: 0.0539, t: 0.1375, p: 21.7030, theta: 59.827, theta_d: 187.841
100% 18/18 [06:01<00:00, 6.99s/it]
18/|/| 18/? [06:01<00:00, 6.99s/it]
100% 18/18 [05:02<00:00, 5.56s/it]
18/|/| 18/? [05:02<00:00, 5.55s/it]
Epoch 71 Loss => total: 40.0160 , g: 18.1274, l: 0.0534, t: 0.1360, p: 21.6993, theta: 58.387, theta_d: 105.327
100% 18/18 [06:04<00:00, 6.33s/it]
18/|/| 18/? [06:04<00:00, 6.33s/it]
100% 18/18 [05:03<00:00, 4.91s/it]
18/|/| 18/? [05:03<00:00, 4.91s/it]
Epoch 72 Loss => total: 18.7964 , g: 17.9851, l: 0.0529, t: 0.1345, p: 0.6239, theta: 61.260, theta_d: 269.911
100% 18/18 [06:07<00:00, 7.04s/it]
18/|/| 18/? [06:07<00:00, 7.04s/it]
100% 18/18 [05:02<00:00, 4.54s/it]
18/|/| 18/? [05:02<00:00, 4.54s/it]
Epoch 73 Loss => total: 18.0195 , g: 17.8269, l: 0.0525, t: 0.1330, p: 0.0071, theta: 57.325, theta_d: 44.485
100% 18/18 [06:03<00:00, 6.17s/it]
18/|/| 18/? [06:03<00:00, 6.17s/it]
100% 18/18 [05:03<00:00, 4.89s/it]
18/|/| 18/? [05:03<00:00, 4.89s/it]
Epoch 74 Loss => total: 17.8823 , g: 17.6966, l: 0.0519, t: 0.1316, p: 0.0021, theta: 57.945, theta_d: 79.999
100% 18/18 [06:03<00:00, 6.36s/it]
18/|/| 18/? [06:03<00:00, 6.36s/it]
100% 18/18 [05:03<00:00, 4.81s/it]
18/|/| 18/? [05:03<00:00, 4.81s/it]
Epoch 75 Loss => total: 39.4375 , g: 17.5571, l: 0.0515, t: 0.1303, p: 21.6987, theta: 59.904, theta_d: 192.230
100% 18/18 [06:05<00:00, 6.36s/it]
18/|/| 18/? [06:05<00:00, 6.36s/it]
100% 18/18 [05:05<00:00, 4.80s/it]
18/|/| 18/? [05:05<00:00, 4.80s/it]
Epoch 76 Loss => total: 17.5843 , g: 17.4030, l: 0.0511, t: 0.1288, p: 0.0014, theta: 61.282, theta_d: 271.197
100% 18/18 [06:06<00:00, 20.35s/it]
18/|/| 18/? [06:06<00:00, 20.35s/it]
100% 18/18 [05:03<00:00, 5.16s/it]
18/|/| 18/? [05:03<00:00, 5.16s/it]
Epoch 77 Loss => total: 39.1261 , g: 17.2556, l: 0.0506, t: 0.1274, p: 21.6926, theta: 60.748, theta_d: 240.627
100% 18/18 [06:09<00:00, 6.26s/it]
18/|/| 18/? [06:09<00:00, 6.26s/it]
100% 18/18 [05:05<00:00, 6.28s/it]
18/|/| 18/? [05:05<00:00, 6.28s/it]
Epoch 78 Loss => total: 38.9773 , g: 17.1107, l: 0.0502, t: 0.1259, p: 21.6905, theta: 60.162, theta_d: 207.054
100% 18/18 [06:21<00:00, 7.43s/it]
18/|/| 18/? [06:21<00:00, 7.43s/it]
100% 18/18 [05:07<00:00, 5.13s/it]
18/|/| 18/? [05:07<00:00, 5.13s/it]
Epoch 79 Loss => total: 38.8298 , g: 16.9673, l: 0.0497, t: 0.1245, p: 21.6883, theta: 59.931, theta_d: 193.778
100% 18/18 [06:25<00:00, 7.19s/it]
18/|/| 18/? [06:25<00:00, 7.19s/it]
100% 18/18 [05:10<00:00, 5.15s/it]
18/|/| 18/? [05:10<00:00, 5.15s/it]
Epoch 80 Loss => total: 38.6844 , g: 16.8255, l: 0.0493, t: 0.1231, p: 21.6865, theta: 59.221, theta_d: 153.121
100% 18/18 [06:24<00:00, 7.20s/it]
18/|/| 18/? [06:24<00:00, 7.20s/it]
100% 18/18 [05:06<00:00, 5.27s/it]
18/|/| 18/? [05:06<00:00, 5.27s/it]
Epoch 81 Loss => total: 16.8740 , g: 16.6947, l: 0.0487, t: 0.1218, p: 0.0088, theta: 61.923, theta_d: 307.898
100% 18/18 [06:26<00:00, 6.80s/it]
18/|/| 18/? [06:26<00:00, 6.80s/it]
100% 18/18 [05:06<00:00, 5.27s/it]
18/|/| 18/? [05:06<00:00, 5.27s/it]
Epoch 82 Loss => total: 16.7359 , g: 16.5656, l: 0.0483, t: 0.1205, p: 0.0015, theta: 61.322, theta_d: 273.472
100% 18/18 [06:21<00:00, 5.41s/it]
18/|/| 18/? [06:21<00:00, 5.41s/it]
100% 18/18 [05:07<00:00, 5.34s/it]
18/|/| 18/? [05:07<00:00, 5.34s/it]
Epoch 83 Loss => total: 16.6018 , g: 16.4302, l: 0.0478, t: 0.1192, p: 0.0047, theta: 62.039, theta_d: 314.596
100% 18/18 [06:21<00:00, 6.59s/it]
18/|/| 18/? [06:20<00:00, 6.59s/it]
100% 18/18 [05:06<00:00, 4.81s/it]
18/|/| 18/? [05:06<00:00, 4.81s/it]
Epoch 84 Loss => total: 16.4728 , g: 16.3039, l: 0.0474, t: 0.1179, p: 0.0036, theta: 63.346, theta_d: 29.457
100% 18/18 [06:22<00:00, 6.15s/it]
18/|/| 18/? [06:22<00:00, 6.15s/it]
100% 18/18 [05:08<00:00, 5.75s/it]
18/|/| 18/? [05:08<00:00, 5.75s/it]
Epoch 85 Loss => total: 38.0310 , g: 16.1847, l: 0.0470, t: 0.1167, p: 21.6826, theta: 65.298, theta_d: 141.282
100% 18/18 [06:20<00:00, 5.88s/it]
18/|/| 18/? [06:20<00:00, 5.88s/it]
100% 18/18 [05:04<00:00, 4.85s/it]
18/|/| 18/? [05:04<00:00, 4.85s/it]
Epoch 86 Loss => total: 16.2242 , g: 16.0597, l: 0.0465, t: 0.1155, p: 0.0025, theta: 67.703, theta_d: 279.119
100% 18/18 [06:19<00:00, 5.92s/it]
18/|/| 18/? [06:19<00:00, 5.92s/it]
100% 18/18 [05:04<00:00, 4.60s/it]
18/|/| 18/? [05:04<00:00, 4.60s/it]
Epoch 87 Loss => total: 37.7738 , g: 15.9325, l: 0.0461, t: 0.1142, p: 21.6810, theta: 66.636, theta_d: 217.957
100% 18/18 [06:13<00:00, 5.90s/it]
18/|/| 18/? [06:13<00:00, 5.90s/it]
100% 18/18 [05:07<00:00, 5.13s/it]
18/|/| 18/? [05:07<00:00, 5.13s/it]
Epoch 88 Loss => total: 37.6403 , g: 15.8032, l: 0.0457, t: 0.1130, p: 21.6785, theta: 65.899, theta_d: 175.716
100% 18/18 [06:20<00:00, 5.55s/it]
18/|/| 18/? [06:19<00:00, 5.55s/it]
100% 18/18 [05:06<00:00, 5.17s/it]
18/|/| 18/? [05:06<00:00, 5.17s/it]
Epoch 89 Loss => total: 37.5126 , g: 15.6783, l: 0.0453, t: 0.1117, p: 21.6772, theta: 66.270, theta_d: 197.007
100% 18/18 [06:12<00:00, 5.73s/it]
18/|/| 18/? [06:12<00:00, 5.73s/it]
100% 18/18 [05:06<00:00, 5.15s/it]
18/|/| 18/? [05:06<00:00, 5.15s/it]
Epoch 90 Loss => total: 37.3804 , g: 15.5513, l: 0.0449, t: 0.1105, p: 21.6738, theta: 67.122, theta_d: 245.790
100% 18/18 [06:10<00:00, 5.78s/it]
18/|/| 18/? [06:10<00:00, 5.78s/it]
100% 18/18 [05:05<00:00, 4.85s/it]
18/|/| 18/? [05:05<00:00, 4.85s/it]
Epoch 91 Loss => total: 15.5943 , g: 15.4263, l: 0.0445, t: 0.1093, p: 0.0142, theta: 63.973, theta_d: 65.403
100% 18/18 [06:15<00:00, 6.29s/it]
18/|/| 18/? [06:14<00:00, 6.29s/it]
100% 18/18 [05:05<00:00, 4.87s/it]
18/|/| 18/? [05:05<00:00, 4.87s/it]
Epoch 92 Loss => total: 37.1504 , g: 15.3234, l: 0.0440, t: 0.1082, p: 21.6747, theta: 65.732, theta_d: 166.171
100% 18/18 [06:11<00:00, 6.35s/it]
18/|/| 18/? [06:11<00:00, 6.35s/it]
100% 18/18 [05:06<00:00, 5.05s/it]
18/|/| 18/? [05:06<00:00, 5.05s/it]
Epoch 93 Loss => total: 37.0249 , g: 15.2036, l: 0.0436, t: 0.1070, p: 21.6707, theta: 66.249, theta_d: 195.771
100% 18/18 [06:14<00:00, 6.15s/it]
18/|/| 18/? [06:14<00:00, 6.15s/it]
100% 18/18 [05:06<00:00, 4.84s/it]
18/|/| 18/? [05:06<00:00, 4.84s/it]
Epoch 94 Loss => total: 36.8988 , g: 15.0810, l: 0.0433, t: 0.1059, p: 21.6686, theta: 65.270, theta_d: 139.669
100% 18/18 [06:11<00:00, 5.84s/it]
18/|/| 18/? [06:11<00:00, 5.84s/it]
100% 18/18 [05:06<00:00, 5.22s/it]
18/|/| 18/? [05:06<00:00, 5.22s/it]
Epoch 95 Loss => total: 36.7845 , g: 14.9714, l: 0.0428, t: 0.1047, p: 21.6656, theta: 64.837, theta_d: 114.901
100% 18/18 [06:07<00:00, 5.25s/it]
18/|/| 18/? [06:07<00:00, 5.25s/it]
100% 18/18 [05:19<00:00, 5.28s/it]
18/|/| 18/? [05:18<00:00, 5.28s/it]
Epoch 96 Loss => total: 36.6697 , g: 14.8589, l: 0.0425, t: 0.1037, p: 21.6647, theta: 66.950, theta_d: 235.928
100% 18/18 [06:16<00:00, 6.39s/it]
18/|/| 18/? [06:16<00:00, 6.39s/it]
100% 18/18 [05:07<00:00, 5.29s/it]
18/|/| 18/? [05:07<00:00, 5.29s/it]
Epoch 97 Loss => total: 36.5497 , g: 14.7416, l: 0.0421, t: 0.1025, p: 21.6635, theta: 65.434, theta_d: 149.073
100% 18/18 [06:19<00:00, 5.94s/it]
18/|/| 18/? [06:19<00:00, 5.94s/it]
100% 18/18 [05:12<00:00, 5.11s/it]
18/|/| 18/? [05:12<00:00, 5.11s/it]
Epoch 98 Loss => total: 36.4385 , g: 14.6352, l: 0.0417, t: 0.1014, p: 21.6601, theta: 65.207, theta_d: 136.097
100% 18/18 [06:23<00:00, 6.90s/it]
18/|/| 18/? [06:23<00:00, 6.90s/it]
100% 18/18 [05:09<00:00, 5.71s/it]
18/|/| 18/? [05:09<00:00, 5.71s/it]
Epoch 99 Loss => total: 36.3261 , g: 14.5280, l: 0.0413, t: 0.1004, p: 21.6564, theta: 67.097, theta_d: 244.394
100% 18/18 [06:17<00:00, 6.64s/it]
18/|/| 18/? [06:17<00:00, 6.64s/it]
100% 18/18 [05:11<00:00, 5.08s/it]
18/|/| 18/? [05:11<00:00, 5.08s/it]
Epoch 100 Loss => total: 36.2125 , g: 14.4149, l: 0.0410, t: 0.0993, p: 21.6574, theta: 65.228, theta_d: 137.310
100% 18/18 [06:33<00:00, 7.20s/it]
18/|/| 18/? [06:33<00:00, 7.20s/it]
100% 18/18 [05:12<00:00, 5.27s/it]
18/|/| 18/? [05:12<00:00, 5.27s/it]
Epoch 101 Loss => total: 36.1082 , g: 14.3134, l: 0.0406, t: 0.0983, p: 21.6559, theta: 65.861, theta_d: 173.578
100% 18/18 [06:43<00:00, 22.42s/it]
18/|/| 18/? [06:43<00:00, 22.41s/it]
100% 18/18 [05:11<00:00, 4.07s/it]
18/|/| 18/? [05:11<00:00, 4.07s/it]
Epoch 102 Loss => total: 35.9487 , g: 14.2059, l: 0.0402, t: 0.0972, p: 21.6054, theta: 64.570, theta_d: 99.595
100% 18/18 [06:55<00:00, 7.21s/it]
18/|/| 18/? [06:55<00:00, 7.21s/it]
100% 18/18 [05:33<00:00, 5.03s/it]
18/|/| 18/? [05:33<00:00, 5.03s/it]
Epoch 103 Loss => total: 15.9470 , g: 14.1028, l: 0.0399, t: 0.0962, p: 1.7081, theta: 64.414, theta_d: 90.631
100% 18/18 [06:44<00:00, 22.46s/it]
18/|/| 18/? [06:44<00:00, 22.46s/it]
100% 18/18 [05:12<00:00, 6.62s/it]
18/|/| 18/? [05:12<00:00, 6.62s/it]
Epoch 104 Loss => total: 35.7573 , g: 13.9956, l: 0.0395, t: 0.0951, p: 21.6271, theta: 65.018, theta_d: 125.236
100% 18/18 [06:28<00:00, 5.43s/it]
18/|/| 18/? [06:28<00:00, 5.43s/it]
100% 18/18 [05:07<00:00, 5.43s/it]
18/|/| 18/? [05:07<00:00, 5.43s/it]
Epoch 105 Loss => total: 35.6618 , g: 13.8920, l: 0.0392, t: 0.0941, p: 21.6365, theta: 65.638, theta_d: 160.757
100% 18/18 [06:27<00:00, 5.25s/it]
18/|/| 18/? [06:27<00:00, 5.25s/it]
100% 18/18 [05:06<00:00, 5.50s/it]
18/|/| 18/? [05:06<00:00, 5.50s/it]
Epoch 106 Loss => total: 35.5481 , g: 13.7885, l: 0.0388, t: 0.0931, p: 21.6277, theta: 65.756, theta_d: 167.530
100% 18/18 [06:26<00:00, 21.49s/it]
18/|/| 18/? [06:26<00:00, 21.49s/it]
100% 18/18 [05:08<00:00, 5.15s/it]
18/|/| 18/? [05:08<00:00, 5.15s/it]
Epoch 107 Loss => total: 35.4286 , g: 13.6856, l: 0.0385, t: 0.0921, p: 21.6123, theta: 65.965, theta_d: 179.517
100% 18/18 [06:21<00:00, 6.52s/it]
18/|/| 18/? [06:21<00:00, 6.52s/it]
100% 18/18 [05:10<00:00, 5.18s/it]
18/|/| 18/? [05:10<00:00, 5.18s/it]
Epoch 108 Loss => total: 14.9874 , g: 13.5768, l: 0.0382, t: 0.0911, p: 1.2813, theta: 65.942, theta_d: 178.199
100% 18/18 [06:19<00:00, 6.10s/it]
18/|/| 18/? [06:19<00:00, 6.10s/it]
100% 18/18 [05:06<00:00, 5.16s/it]
18/|/| 18/? [05:06<00:00, 5.16s/it]
Epoch 109 Loss => total: 14.8650 , g: 13.4590, l: 0.0377, t: 0.0899, p: 1.2784, theta: 65.941, theta_d: 178.152
100% 18/18 [06:19<00:00, 6.21s/it]
18/|/| 18/? [06:19<00:00, 6.21s/it]
100% 18/18 [05:05<00:00, 6.13s/it]
18/|/| 18/? [05:05<00:00, 6.13s/it]
Epoch 110 Loss => total: 14.7426 , g: 13.3383, l: 0.0373, t: 0.0887, p: 1.2783, theta: 65.941, theta_d: 178.113
100% 18/18 [06:14<00:00, 5.67s/it]
18/|/| 18/? [06:14<00:00, 5.67s/it]
100% 18/18 [05:05<00:00, 4.71s/it]
18/|/| 18/? [05:05<00:00, 4.71s/it]
Epoch 111 Loss => total: 14.6197 , g: 13.2169, l: 0.0369, t: 0.0875, p: 1.2784, theta: 65.940, theta_d: 178.063
100% 18/18 [06:15<00:00, 6.06s/it]
18/|/| 18/? [06:15<00:00, 6.06s/it]
100% 18/18 [05:04<00:00, 5.06s/it]
18/|/| 18/? [05:04<00:00, 5.06s/it]
Epoch 112 Loss => total: 14.4957 , g: 13.0944, l: 0.0364, t: 0.0863, p: 1.2785, theta: 65.938, theta_d: 177.993
100% 18/18 [06:11<00:00, 5.99s/it]
18/|/| 18/? [06:11<00:00, 5.99s/it]
100% 18/18 [05:05<00:00, 5.04s/it]
18/|/| 18/? [05:05<00:00, 5.04s/it]
Epoch 113 Loss => total: 14.3695 , g: 12.9698, l: 0.0360, t: 0.0851, p: 1.2786, theta: 65.937, theta_d: 177.896
100% 18/18 [06:08<00:00, 5.67s/it]
18/|/| 18/? [06:08<00:00, 5.67s/it]
100% 18/18 [05:03<00:00, 5.39s/it]
18/|/| 18/? [05:03<00:00, 5.39s/it]
Epoch 114 Loss => total: 14.2416 , g: 12.8435, l: 0.0355, t: 0.0839, p: 1.2787, theta: 65.934, theta_d: 177.763
100% 18/18 [06:06<00:00, 6.23s/it]
18/|/| 18/? [06:06<00:00, 6.23s/it]
100% 18/18 [05:03<00:00, 5.22s/it]
18/|/| 18/? [05:03<00:00, 5.22s/it]
Epoch 115 Loss => total: 14.1120 , g: 12.7155, l: 0.0351, t: 0.0826, p: 1.2788, theta: 65.931, theta_d: 177.582
100% 18/18 [06:02<00:00, 5.97s/it]
18/|/| 18/? [06:02<00:00, 5.97s/it]
100% 18/18 [05:05<00:00, 5.46s/it]
18/|/| 18/? [05:05<00:00, 5.46s/it]
Epoch 116 Loss => total: 13.9806 , g: 12.5858, l: 0.0346, t: 0.0813, p: 1.2789, theta: 65.927, theta_d: 177.331
100% 18/18 [06:25<00:00, 7.65s/it]
18/|/| 18/? [06:25<00:00, 7.65s/it]
100% 18/18 [05:07<00:00, 5.01s/it]
18/|/| 18/? [05:07<00:00, 5.01s/it]
Epoch 117 Loss => total: 13.8542 , g: 12.4611, l: 0.0341, t: 0.0801, p: 1.2790, theta: 65.921, theta_d: 176.980
100% 18/18 [06:10<00:00, 5.79s/it]
18/|/| 18/? [06:10<00:00, 5.79s/it]
100% 18/18 [05:03<00:00, 5.24s/it]
18/|/| 18/? [05:03<00:00, 5.24s/it]
Epoch 118 Loss => total: 13.7576 , g: 12.3656, l: 0.0338, t: 0.0791, p: 1.2791, theta: 65.912, theta_d: 176.471
100% 18/18 [06:10<00:00, 6.24s/it]
18/|/| 18/? [06:10<00:00, 6.24s/it]
100% 18/18 [05:03<00:00, 5.96s/it]
18/|/| 18/? [05:03<00:00, 5.96s/it]
Epoch 119 Loss => total: 13.6779 , g: 12.2868, l: 0.0336, t: 0.0784, p: 1.2791, theta: 65.899, theta_d: 175.725
100% 18/18 [06:10<00:00, 6.67s/it]
18/|/| 18/? [06:10<00:00, 6.67s/it]
100% 18/18 [05:08<00:00, 5.58s/it]
18/|/| 18/? [05:08<00:00, 5.58s/it]
Epoch 120 Loss => total: 13.6003 , g: 12.2102, l: 0.0333, t: 0.0777, p: 1.2792, theta: 65.878, theta_d: 174.528
100% 18/18 [06:07<00:00, 6.83s/it]
18/|/| 18/? [06:07<00:00, 6.83s/it]
100% 18/18 [05:02<00:00, 4.76s/it]
18/|/| 18/? [05:02<00:00, 4.76s/it]
Epoch 121 Loss => total: 13.5231 , g: 12.1340, l: 0.0331, t: 0.0769, p: 1.2791, theta: 65.842, theta_d: 172.446
100% 18/18 [06:05<00:00, 7.94s/it]
18/|/| 18/? [06:04<00:00, 7.94s/it]
100% 18/18 [05:04<00:00, 16.92s/it]
18/|/| 18/? [05:04<00:00, 16.92s/it]
Epoch 122 Loss => total: 13.4472 , g: 12.0592, l: 0.0328, t: 0.0762, p: 1.2789, theta: 65.778, theta_d: 168.776
100% 18/18 [06:05<00:00, 6.42s/it]
18/|/| 18/? [06:05<00:00, 6.42s/it]
100% 18/18 [05:02<00:00, 4.88s/it]
18/|/| 18/? [05:02<00:00, 4.88s/it]
Epoch 123 Loss => total: 13.3729 , g: 11.9860, l: 0.0326, t: 0.0755, p: 1.2788, theta: 65.668, theta_d: 162.514
100% 18/18 [06:09<00:00, 6.03s/it]
18/|/| 18/? [06:09<00:00, 6.03s/it]
100% 18/18 [05:02<00:00, 4.62s/it]
18/|/| 18/? [05:02<00:00, 4.62s/it]
Epoch 124 Loss => total: 13.2981 , g: 11.9139, l: 0.0323, t: 0.0749, p: 1.2770, theta: 65.557, theta_d: 156.159
100% 18/18 [06:03<00:00, 6.03s/it]
18/|/| 18/? [06:03<00:00, 6.03s/it]
100% 18/18 [05:02<00:00, 4.89s/it]
18/|/| 18/? [05:02<00:00, 4.89s/it]
Epoch 125 Loss => total: 13.2269 , g: 11.8437, l: 0.0321, t: 0.0742, p: 1.2769, theta: 65.404, theta_d: 147.382
100% 18/18 [06:02<00:00, 6.11s/it]
18/|/| 18/? [06:02<00:00, 6.11s/it]
100% 18/18 [05:02<00:00, 4.80s/it]
18/|/| 18/? [05:02<00:00, 4.79s/it]
Epoch 126 Loss => total: 13.1571 , g: 11.7748, l: 0.0318, t: 0.0735, p: 1.2770, theta: 65.254, theta_d: 138.776
100% 18/18 [06:00<00:00, 5.86s/it]
18/|/| 18/? [06:00<00:00, 5.86s/it]
100% 18/18 [05:03<00:00, 4.98s/it]
18/|/| 18/? [05:03<00:00, 4.98s/it]
Epoch 127 Loss => total: 13.0873 , g: 11.7059, l: 0.0315, t: 0.0728, p: 1.2770, theta: 65.118, theta_d: 130.978
100% 18/18 [05:59<00:00, 5.76s/it]
18/|/| 18/? [05:59<00:00, 5.76s/it]
100% 18/18 [05:03<00:00, 4.79s/it]
18/|/| 18/? [05:03<00:00, 4.79s/it]
Epoch 128 Loss => total: 13.0175 , g: 11.6371, l: 0.0313, t: 0.0722, p: 1.2769, theta: 64.991, theta_d: 123.721
100% 18/18 [06:04<00:00, 6.37s/it]
18/|/| 18/? [06:04<00:00, 6.37s/it]
100% 18/18 [05:02<00:00, 4.73s/it]
18/|/| 18/? [05:02<00:00, 4.73s/it]
Epoch 129 Loss => total: 12.9475 , g: 11.5682, l: 0.0310, t: 0.0715, p: 1.2768, theta: 64.872, theta_d: 116.872
100% 18/18 [06:02<00:00, 6.14s/it]
18/|/| 18/? [06:02<00:00, 6.14s/it]
100% 18/18 [05:14<00:00, 5.07s/it]
18/|/| 18/? [05:14<00:00, 5.07s/it]
Epoch 130 Loss => total: 12.8774 , g: 11.4992, l: 0.0307, t: 0.0708, p: 1.2766, theta: 64.758, theta_d: 110.382
100% 18/18 [06:01<00:00, 6.22s/it]
18/|/| 18/? [06:01<00:00, 6.22s/it]
100% 18/18 [05:04<00:00, 4.90s/it]
18/|/| 18/? [05:04<00:00, 4.90s/it]
Epoch 131 Loss => total: 12.8067 , g: 11.4300, l: 0.0304, t: 0.0701, p: 1.2762, theta: 64.651, theta_d: 104.241
100% 18/18 [06:39<00:00, 6.06s/it]
18/|/| 18/? [06:39<00:00, 6.06s/it]
100% 18/18 [05:03<00:00, 4.72s/it]
18/|/| 18/? [05:03<00:00, 4.72s/it]
Epoch 132 Loss => total: 12.7342 , g: 11.3595, l: 0.0302, t: 0.0694, p: 1.2750, theta: 64.550, theta_d: 98.451
100% 18/18 [06:03<00:00, 6.24s/it]
18/|/| 18/? [06:03<00:00, 6.24s/it]
100% 18/18 [05:03<00:00, 4.69s/it]
18/|/| 18/? [05:03<00:00, 4.69s/it]
Epoch 133 Loss => total: 12.6616 , g: 11.2896, l: 0.0299, t: 0.0687, p: 1.2734, theta: 64.455, theta_d: 93.021
100% 18/18 [06:01<00:00, 20.08s/it]
18/|/| 18/? [06:01<00:00, 20.08s/it]
100% 18/18 [05:03<00:00, 4.88s/it]
18/|/| 18/? [05:03<00:00, 4.88s/it]
Epoch 134 Loss => total: 27.2327 , g: 11.2185, l: 0.0296, t: 0.0681, p: 15.9165, theta: 64.367, theta_d: 87.966
100% 18/18 [06:01<00:00, 6.33s/it]
18/|/| 18/? [06:01<00:00, 6.33s/it]
100% 18/18 [05:06<00:00, 5.12s/it]
18/|/| 18/? [05:06<00:00, 5.12s/it]
Epoch 135 Loss => total: 33.9657 , g: 11.1163, l: 0.0293, t: 0.0671, p: 22.7531, theta: 64.287, theta_d: 83.384
100% 18/18 [06:00<00:00, 7.60s/it]
18/|/| 18/? [06:00<00:00, 7.60s/it]
100% 18/18 [05:04<00:00, 6.68s/it]
18/|/| 18/? [05:04<00:00, 6.68s/it]
Epoch 136 Loss => total: 33.9430 , g: 11.0410, l: 0.0291, t: 0.0664, p: 22.8066, theta: 64.227, theta_d: 79.936
100% 18/18 [06:01<00:00, 6.36s/it]
18/|/| 18/? [06:01<00:00, 6.36s/it]
100% 18/18 [05:04<00:00, 5.00s/it]
18/|/| 18/? [05:04<00:00, 5.00s/it]
Epoch 137 Loss => total: 33.8791 , g: 10.9675, l: 0.0289, t: 0.0657, p: 22.8170, theta: 64.173, theta_d: 76.838
100% 18/18 [06:03<00:00, 6.99s/it]
18/|/| 18/? [06:03<00:00, 6.99s/it]
100% 18/18 [05:03<00:00, 4.67s/it]
18/|/| 18/? [05:03<00:00, 4.67s/it]
Epoch 138 Loss => total: 33.8078 , g: 10.8966, l: 0.0287, t: 0.0651, p: 22.8175, theta: 64.123, theta_d: 74.004
100% 18/18 [06:02<00:00, 5.68s/it]
18/|/| 18/? [06:02<00:00, 5.68s/it]
100% 18/18 [05:04<00:00, 4.88s/it]
18/|/| 18/? [05:04<00:00, 4.88s/it]
Epoch 139 Loss => total: 33.7317 , g: 10.8256, l: 0.0285, t: 0.0644, p: 22.8131, theta: 64.078, theta_d: 71.420
100% 18/18 [06:01<00:00, 5.89s/it]
18/|/| 18/? [06:01<00:00, 5.89s/it]
100% 18/18 [05:06<00:00, 4.96s/it]
18/|/| 18/? [05:06<00:00, 4.96s/it]
Epoch 140 Loss => total: 33.6553 , g: 10.7570, l: 0.0282, t: 0.0638, p: 22.8063, theta: 64.037, theta_d: 69.068
100% 18/18 [06:02<00:00, 6.17s/it]
18/|/| 18/? [06:02<00:00, 6.17s/it]
100% 18/18 [05:02<00:00, 4.69s/it]
18/|/| 18/? [05:02<00:00, 4.69s/it]
Epoch 141 Loss => total: 33.5766 , g: 10.6883, l: 0.0280, t: 0.0632, p: 22.7971, theta: 64.000, theta_d: 66.917
100% 18/18 [06:01<00:00, 6.46s/it]
18/|/| 18/? [06:01<00:00, 6.46s/it]
100% 18/18 [05:02<00:00, 4.40s/it]
18/|/| 18/? [05:02<00:00, 4.40s/it]
Epoch 142 Loss => total: 33.4986 , g: 10.6214, l: 0.0278, t: 0.0625, p: 22.7869, theta: 63.966, theta_d: 64.974
100% 18/18 [06:02<00:00, 6.15s/it]
18/|/| 18/? [06:02<00:00, 6.15s/it]
100% 18/18 [05:04<00:00, 4.54s/it]
18/|/| 18/? [05:04<00:00, 4.54s/it]
Epoch 143 Loss => total: 33.4215 , g: 10.5553, l: 0.0276, t: 0.0619, p: 22.7767, theta: 63.935, theta_d: 63.182
100% 18/18 [06:03<00:00, 5.60s/it]
18/|/| 18/? [06:03<00:00, 5.60s/it]
100% 18/18 [05:01<00:00, 4.55s/it]
18/|/| 18/? [05:01<00:00, 4.55s/it]
Epoch 144 Loss => total: 33.3466 , g: 10.4899, l: 0.0274, t: 0.0613, p: 22.7680, theta: 63.906, theta_d: 61.533
100% 18/18 [06:02<00:00, 7.55s/it]
18/|/| 18/? [06:02<00:00, 7.55s/it]
100% 18/18 [05:03<00:00, 4.93s/it]
18/|/| 18/? [05:03<00:00, 4.93s/it]
Epoch 145 Loss => total: 33.2770 , g: 10.4263, l: 0.0272, t: 0.0607, p: 22.7627, theta: 63.880, theta_d: 60.049
100% 18/18 [06:01<00:00, 6.37s/it]
18/|/| 18/? [06:01<00:00, 6.37s/it]
100% 18/18 [05:04<00:00, 6.06s/it]
18/|/| 18/? [05:04<00:00, 6.06s/it]
Epoch 146 Loss => total: 33.2123 , g: 10.3652, l: 0.0269, t: 0.0601, p: 22.7600, theta: 63.856, theta_d: 58.697
100% 18/18 [06:01<00:00, 5.80s/it]
18/|/| 18/? [06:01<00:00, 5.80s/it]
100% 18/18 [05:05<00:00, 5.57s/it]
18/|/| 18/? [05:05<00:00, 5.57s/it]
Epoch 147 Loss => total: 33.1485 , g: 10.3039, l: 0.0267, t: 0.0596, p: 22.7583, theta: 63.835, theta_d: 57.459
100% 18/18 [06:04<00:00, 6.56s/it]
18/|/| 18/? [06:04<00:00, 6.56s/it]
100% 18/18 [05:02<00:00, 5.01s/it]
18/|/| 18/? [05:02<00:00, 5.01s/it]
Epoch 148 Loss => total: 33.0840 , g: 10.2419, l: 0.0265, t: 0.0590, p: 22.7567, theta: 63.814, theta_d: 56.290
100% 18/18 [06:02<00:00, 5.98s/it]
18/|/| 18/? [06:02<00:00, 5.98s/it]
100% 18/18 [05:04<00:00, 5.19s/it]
18/|/| 18/? [05:04<00:00, 5.19s/it]
Epoch 149 Loss => total: 33.0210 , g: 10.1815, l: 0.0263, t: 0.0584, p: 22.7548, theta: 63.795, theta_d: 55.160
100% 18/18 [06:02<00:00, 5.80s/it]
18/|/| 18/? [06:02<00:00, 5.80s/it]
100% 18/18 [05:04<00:00, 4.67s/it]
18/|/| 18/? [05:04<00:00, 4.67s/it]
Epoch 150 Loss => total: 32.9582 , g: 10.1218, l: 0.0261, t: 0.0578, p: 22.7526, theta: 63.777, theta_d: 54.169
100% 18/18 [06:02<00:00, 6.15s/it]
18/|/| 18/? [06:02<00:00, 6.15s/it]
100% 18/18 [05:03<00:00, 4.82s/it]
18/|/| 18/? [05:03<00:00, 4.82s/it]
Epoch 151 Loss => total: 32.8953 , g: 10.0624, l: 0.0258, t: 0.0573, p: 22.7498, theta: 63.759, theta_d: 53.140
100% 18/18 [06:03<00:00, 6.25s/it]
18/|/| 18/? [06:03<00:00, 6.25s/it]
100% 18/18 [05:02<00:00, 4.62s/it]
18/|/| 18/? [05:02<00:00, 4.62s/it]
Epoch 152 Loss => total: 32.8318 , g: 10.0031, l: 0.0256, t: 0.0567, p: 22.7464, theta: 63.743, theta_d: 52.205
100% 18/18 [06:02<00:00, 6.25s/it]
18/|/| 18/? [06:02<00:00, 6.25s/it]
100% 18/18 [05:04<00:00, 16.92s/it]
18/|/| 18/? [05:04<00:00, 16.91s/it]
Epoch 153 Loss => total: 32.7681 , g: 9.9442, l: 0.0254, t: 0.0562, p: 22.7423, theta: 63.727, theta_d: 51.300
100% 18/18 [06:01<00:00, 6.00s/it]
18/|/| 18/? [06:01<00:00, 6.00s/it]
100% 18/18 [05:04<00:00, 4.41s/it]
18/|/| 18/? [05:04<00:00, 4.41s/it]
Epoch 154 Loss => total: 32.7058 , g: 9.8877, l: 0.0252, t: 0.0556, p: 22.7372, theta: 63.711, theta_d: 50.392
100% 18/18 [06:04<00:00, 6.67s/it]
18/|/| 18/? [06:04<00:00, 6.67s/it]
100% 18/18 [05:04<00:00, 4.83s/it]
18/|/| 18/? [05:04<00:00, 4.83s/it]
Epoch 155 Loss => total: 32.6412 , g: 9.8301, l: 0.0250, t: 0.0551, p: 22.7310, theta: 63.696, theta_d: 49.485
100% 18/18 [06:03<00:00, 7.38s/it]
18/|/| 18/? [06:03<00:00, 7.38s/it]
100% 18/18 [05:03<00:00, 4.86s/it]
18/|/| 18/? [05:03<00:00, 4.86s/it]
Epoch 156 Loss => total: 32.5712 , g: 9.7681, l: 0.0248, t: 0.0546, p: 22.7237, theta: 63.680, theta_d: 48.570
100% 18/18 [06:03<00:00, 6.58s/it]
18/|/| 18/? [06:03<00:00, 6.58s/it]
100% 18/18 [05:03<00:00, 6.60s/it]
18/|/| 18/? [05:03<00:00, 6.60s/it]
Epoch 157 Loss => total: 32.5034 , g: 9.7097, l: 0.0246, t: 0.0541, p: 22.7151, theta: 63.662, theta_d: 47.580
100% 18/18 [06:02<00:00, 6.32s/it]
18/|/| 18/? [06:02<00:00, 6.32s/it]
100% 18/18 [05:02<00:00, 4.72s/it]
18/|/| 18/? [05:02<00:00, 4.72s/it]
Epoch 158 Loss => total: 32.4336 , g: 9.6509, l: 0.0244, t: 0.0536, p: 22.7047, theta: 63.643, theta_d: 46.484
100% 18/18 [06:05<00:00, 6.38s/it]
18/|/| 18/? [06:05<00:00, 6.38s/it]
100% 18/18 [05:02<00:00, 4.53s/it]
18/|/| 18/? [05:02<00:00, 4.53s/it]
Epoch 159 Loss => total: 32.3604 , g: 9.5916, l: 0.0242, t: 0.0530, p: 22.6915, theta: 63.620, theta_d: 45.177
100% 18/18 [06:01<00:00, 5.54s/it]
18/|/| 18/? [06:01<00:00, 5.54s/it]
100% 18/18 [05:03<00:00, 5.17s/it]
18/|/| 18/? [05:03<00:00, 5.17s/it]
Epoch 160 Loss => total: 32.2784 , g: 9.5312, l: 0.0241, t: 0.0525, p: 22.6705, theta: 63.594, theta_d: 43.669
100% 18/18 [06:05<00:00, 8.00s/it]
18/|/| 18/? [06:05<00:00, 8.00s/it]
100% 18/18 [05:02<00:00, 4.74s/it]
18/|/| 18/? [05:02<00:00, 4.74s/it]
Epoch 161 Loss => total: 9.5399 , g: 9.4616, l: 0.0239, t: 0.0520, p: 0.0024, theta: 63.556, theta_d: 41.508
100% 18/18 [06:02<00:00, 5.44s/it]
18/|/| 18/? [06:02<00:00, 5.44s/it]
100% 18/18 [05:04<00:00, 4.88s/it]
18/|/| 18/? [05:04<00:00, 4.88s/it]
Epoch 162 Loss => total: 9.4901 , g: 9.4082, l: 0.0236, t: 0.0515, p: 0.0067, theta: 63.733, theta_d: 51.628
100% 18/18 [06:04<00:00, 6.47s/it]
18/|/| 18/? [06:04<00:00, 6.47s/it]
100% 18/18 [05:06<00:00, 4.88s/it]
18/|/| 18/? [05:06<00:00, 4.88s/it]
Epoch 163 Loss => total: 9.4394 , g: 9.3568, l: 0.0234, t: 0.0510, p: 0.0081, theta: 63.857, theta_d: 58.737
100% 18/18 [06:03<00:00, 20.17s/it]
18/|/| 18/? [06:03<00:00, 20.17s/it]
100% 18/18 [05:02<00:00, 5.06s/it]
18/|/| 18/? [05:02<00:00, 5.06s/it]
Epoch 164 Loss => total: 9.3866 , g: 9.3050, l: 0.0231, t: 0.0506, p: 0.0079, theta: 63.967, theta_d: 65.041
100% 18/18 [06:02<00:00, 6.23s/it]
18/|/| 18/? [06:02<00:00, 6.23s/it]
100% 18/18 [05:06<00:00, 5.78s/it]
18/|/| 18/? [05:06<00:00, 5.78s/it]
Epoch 165 Loss => total: 9.3338 , g: 9.2536, l: 0.0229, t: 0.0501, p: 0.0072, theta: 64.082, theta_d: 71.631
100% 18/18 [06:01<00:00, 6.08s/it]
18/|/| 18/? [06:01<00:00, 6.08s/it]
100% 18/18 [05:03<00:00, 5.06s/it]
18/|/| 18/? [05:03<00:00, 5.06s/it]
Epoch 166 Loss => total: 9.2853 , g: 9.2065, l: 0.0227, t: 0.0497, p: 0.0065, theta: 64.190, theta_d: 77.797
100% 18/18 [06:03<00:00, 7.39s/it]
18/|/| 18/? [06:03<00:00, 7.39s/it]
100% 18/18 [05:02<00:00, 4.69s/it]
18/|/| 18/? [05:02<00:00, 4.69s/it]
Epoch 167 Loss => total: 9.2263 , g: 9.1490, l: 0.0225, t: 0.0492, p: 0.0056, theta: 64.296, theta_d: 83.872
100% 18/18 [06:07<00:00, 6.02s/it]
18/|/| 18/? [06:07<00:00, 6.02s/it]
100% 18/18 [05:04<00:00, 4.91s/it]
18/|/| 18/? [05:04<00:00, 4.91s/it]
Epoch 168 Loss => total: 9.5644 , g: 9.0943, l: 0.0223, t: 0.0488, p: 0.3991, theta: 64.410, theta_d: 90.431
100% 18/18 [06:02<00:00, 5.95s/it]
18/|/| 18/? [06:02<00:00, 5.95s/it]
100% 18/18 [05:04<00:00, 4.75s/it]
18/|/| 18/? [05:04<00:00, 4.74s/it]
Epoch 169 Loss => total: 12.5594 , g: 9.0363, l: 0.0221, t: 0.0483, p: 3.4527, theta: 64.522, theta_d: 96.865
100% 18/18 [06:03<00:00, 5.74s/it]
18/|/| 18/? [06:03<00:00, 5.74s/it]
100% 18/18 [05:04<00:00, 5.17s/it]
18/|/| 18/? [05:04<00:00, 5.17s/it]
Epoch 170 Loss => total: 10.3113 , g: 8.9781, l: 0.0219, t: 0.0478, p: 1.2634, theta: 64.602, theta_d: 101.441
100% 18/18 [06:01<00:00, 5.46s/it]
18/|/| 18/? [06:01<00:00, 5.46s/it]
100% 18/18 [05:04<00:00, 16.90s/it]
18/|/| 18/? [05:04<00:00, 16.89s/it]
Epoch 171 Loss => total: 10.2434 , g: 8.9116, l: 0.0217, t: 0.0474, p: 1.2626, theta: 64.507, theta_d: 95.985
100% 18/18 [06:03<00:00, 6.11s/it]
18/|/| 18/? [06:03<00:00, 6.11s/it]
100% 18/18 [05:05<00:00, 4.85s/it]
18/|/| 18/? [05:05<00:00, 4.85s/it]
Epoch 172 Loss => total: 10.2587 , g: 8.8504, l: 0.0216, t: 0.0470, p: 1.3398, theta: 64.416, theta_d: 90.780
100% 18/18 [06:02<00:00, 7.83s/it]
18/|/| 18/? [06:02<00:00, 7.83s/it]
100% 18/18 [05:03<00:00, 4.74s/it]
18/|/| 18/? [05:03<00:00, 4.74s/it]
Epoch 173 Loss => total: 14.0133 , g: 8.7893, l: 0.0214, t: 0.0465, p: 5.1561, theta: 64.331, theta_d: 85.899
100% 18/18 [06:37<00:00, 6.89s/it]
18/|/| 18/? [06:37<00:00, 6.89s/it]
100% 18/18 [05:04<00:00, 6.40s/it]
18/|/| 18/? [05:04<00:00, 6.40s/it]
Epoch 174 Loss => total: 8.7971 , g: 8.7249, l: 0.0212, t: 0.0461, p: 0.0049, theta: 64.247, theta_d: 81.094
100% 18/18 [06:03<00:00, 5.84s/it]
18/|/| 18/? [06:03<00:00, 5.84s/it]
100% 18/18 [05:02<00:00, 4.81s/it]
18/|/| 18/? [05:02<00:00, 4.81s/it]
Epoch 175 Loss => total: 8.7397 , g: 8.6666, l: 0.0210, t: 0.0457, p: 0.0064, theta: 64.332, theta_d: 85.964
100% 18/18 [06:02<00:00, 5.87s/it]
18/|/| 18/? [06:02<00:00, 5.87s/it]
100% 18/18 [05:03<00:00, 4.57s/it]
18/|/| 18/? [05:03<00:00, 4.57s/it]
Epoch 176 Loss => total: 9.1255 , g: 8.6082, l: 0.0208, t: 0.0453, p: 0.4513, theta: 64.414, theta_d: 90.654
100% 18/18 [06:05<00:00, 6.30s/it]
18/|/| 18/? [06:05<00:00, 6.30s/it]
100% 18/18 [05:03<00:00, 5.11s/it]
18/|/| 18/? [05:03<00:00, 5.11s/it]
Epoch 177 Loss => total: 9.6604 , g: 8.5460, l: 0.0207, t: 0.0448, p: 1.0488, theta: 64.483, theta_d: 94.616
100% 18/18 [06:06<00:00, 8.07s/it]
18/|/| 18/? [06:06<00:00, 8.07s/it]
100% 18/18 [05:04<00:00, 5.07s/it]
18/|/| 18/? [05:04<00:00, 5.07s/it]
Epoch 178 Loss => total: 9.8000 , g: 8.4785, l: 0.0205, t: 0.0444, p: 1.2566, theta: 64.567, theta_d: 99.390
100% 18/18 [06:05<00:00, 6.10s/it]
18/|/| 18/? [06:05<00:00, 6.10s/it]
100% 18/18 [05:03<00:00, 4.92s/it]
18/|/| 18/? [05:03<00:00, 4.92s/it]
Epoch 179 Loss => total: 9.8304 , g: 8.4115, l: 0.0204, t: 0.0440, p: 1.3545, theta: 64.415, theta_d: 90.723
100% 18/18 [06:03<00:00, 6.04s/it]
18/|/| 18/? [06:03<00:00, 6.04s/it]
100% 18/18 [05:03<00:00, 5.24s/it]
18/|/| 18/? [05:03<00:00, 5.24s/it]
Epoch 180 Loss => total: 15.2726 , g: 8.3442, l: 0.0202, t: 0.0436, p: 6.8645, theta: 64.286, theta_d: 83.325
100% 18/18 [06:04<00:00, 7.46s/it]
18/|/| 18/? [06:04<00:00, 7.46s/it]
100% 18/18 [05:02<00:00, 4.60s/it]
18/|/| 18/? [05:02<00:00, 4.60s/it]
Epoch 181 Loss => total: 8.3518 , g: 8.2837, l: 0.0201, t: 0.0432, p: 0.0049, theta: 64.204, theta_d: 78.620
100% 18/18 [06:02<00:00, 6.02s/it]
18/|/| 18/? [06:02<00:00, 6.02s/it]
100% 18/18 [05:03<00:00, 6.21s/it]
18/|/| 18/? [05:03<00:00, 6.21s/it]
Epoch 182 Loss => total: 8.2810 , g: 8.2120, l: 0.0199, t: 0.0428, p: 0.0063, theta: 64.302, theta_d: 84.213
100% 18/18 [06:02<00:00, 6.10s/it]
18/|/| 18/? [06:02<00:00, 6.10s/it]
100% 18/18 [05:02<00:00, 5.52s/it]
18/|/| 18/? [05:02<00:00, 5.52s/it]
Epoch 183 Loss => total: 8.3476 , g: 8.1480, l: 0.0197, t: 0.0424, p: 0.1374, theta: 64.393, theta_d: 89.422
100% 18/18 [06:02<00:00, 6.57s/it]
18/|/| 18/? [06:02<00:00, 6.57s/it]
100% 18/18 [05:03<00:00, 4.81s/it]
18/|/| 18/? [05:03<00:00, 4.81s/it]
Epoch 184 Loss => total: 11.1189 , g: 8.0920, l: 0.0195, t: 0.0421, p: 2.9654, theta: 64.477, theta_d: 94.275
100% 18/18 [06:01<00:00, 6.02s/it]
18/|/| 18/? [06:01<00:00, 6.02s/it]
100% 18/18 [05:02<00:00, 4.47s/it]
18/|/| 18/? [05:02<00:00, 4.47s/it]
Epoch 185 Loss => total: 9.2945 , g: 7.9979, l: 0.0194, t: 0.0416, p: 1.2356, theta: 64.548, theta_d: 98.320
100% 18/18 [06:00<00:00, 5.96s/it]
18/|/| 18/? [06:00<00:00, 5.96s/it]
100% 18/18 [05:02<00:00, 4.86s/it]
18/|/| 18/? [05:02<00:00, 4.86s/it]
Epoch 186 Loss => total: 9.2211 , g: 7.9219, l: 0.0192, t: 0.0412, p: 1.2387, theta: 66.348, theta_d: 201.446
100% 18/18 [06:02<00:00, 6.32s/it]
18/|/| 18/? [06:02<00:00, 6.32s/it]
100% 18/18 [05:02<00:00, 5.04s/it]
18/|/| 18/? [05:02<00:00, 5.04s/it]
Epoch 187 Loss => total: 20.5874 , g: 7.8466, l: 0.0190, t: 0.0409, p: 12.6809, theta: 67.842, theta_d: 287.065
100% 18/18 [06:01<00:00, 6.09s/it]
18/|/| 18/? [06:01<00:00, 6.09s/it]
100% 18/18 [05:01<00:00, 4.69s/it]
18/|/| 18/? [05:01<00:00, 4.69s/it]
Epoch 188 Loss => total: 7.8315 , g: 7.7630, l: 0.0189, t: 0.0405, p: 0.0092, theta: 67.864, theta_d: 288.344
100% 18/18 [06:02<00:00, 6.47s/it]
18/|/| 18/? [06:02<00:00, 6.47s/it]
100% 18/18 [05:01<00:00, 4.80s/it]
18/|/| 18/? [05:01<00:00, 4.80s/it]
Epoch 189 Loss => total: 7.7616 , g: 7.6940, l: 0.0187, t: 0.0401, p: 0.0088, theta: 68.359, theta_d: 316.659
100% 18/18 [06:02<00:00, 6.04s/it]
18/|/| 18/? [06:02<00:00, 6.04s/it]
100% 18/18 [05:05<00:00, 5.46s/it]
18/|/| 18/? [05:05<00:00, 5.46s/it]
Epoch 190 Loss => total: 7.6948 , g: 7.6278, l: 0.0186, t: 0.0398, p: 0.0086, theta: 68.740, theta_d: 338.508
100% 18/18 [05:59<00:00, 6.06s/it]
18/|/| 18/? [05:59<00:00, 6.06s/it]
100% 18/18 [05:06<00:00, 4.85s/it]
18/|/| 18/? [05:05<00:00, 4.85s/it]
Epoch 191 Loss => total: 7.6199 , g: 7.5534, l: 0.0185, t: 0.0394, p: 0.0086, theta: 68.572, theta_d: 328.878
100% 18/18 [06:00<00:00, 6.14s/it]
18/|/| 18/? [06:00<00:00, 6.14s/it]
100% 18/18 [05:01<00:00, 4.76s/it]
18/|/| 18/? [05:01<00:00, 4.76s/it]
Epoch 192 Loss => total: 7.5486 , g: 7.4828, l: 0.0183, t: 0.0391, p: 0.0083, theta: 68.528, theta_d: 326.378
100% 18/18 [05:59<00:00, 7.42s/it]
18/|/| 18/? [05:59<00:00, 7.42s/it]
100% 18/18 [05:00<00:00, 4.58s/it]
18/|/| 18/? [05:00<00:00, 4.58s/it]
Epoch 193 Loss => total: 7.4804 , g: 7.4151, l: 0.0182, t: 0.0388, p: 0.0083, theta: 69.044, theta_d: 355.904
100% 18/18 [05:59<00:00, 5.81s/it]
18/|/| 18/? [05:59<00:00, 5.81s/it]
100% 18/18 [05:03<00:00, 6.30s/it]
18/|/| 18/? [05:03<00:00, 6.30s/it]
Epoch 194 Loss => total: 7.3450 , g: 7.2805, l: 0.0179, t: 0.0382, p: 0.0083, theta: 69.190, theta_d: 4.313
100% 18/18 [05:57<00:00, 19.86s/it]
18/|/| 18/? [05:57<00:00, 19.86s/it]
100% 18/18 [05:14<00:00, 5.57s/it]
18/|/| 18/? [05:14<00:00, 5.57s/it]
Epoch 195 Loss => total: 7.2942 , g: 7.2301, l: 0.0178, t: 0.0379, p: 0.0084, theta: 69.195, theta_d: 4.579
100% 18/18 [06:03<00:00, 5.89s/it]
18/|/| 18/? [06:03<00:00, 5.89s/it]
100% 18/18 [05:03<00:00, 5.06s/it]
18/|/| 18/? [05:03<00:00, 5.06s/it]
Epoch 196 Loss => total: 7.2163 , g: 7.1526, l: 0.0176, t: 0.0376, p: 0.0084, theta: 69.513, theta_d: 22.780
100% 18/18 [06:01<00:00, 7.16s/it]
18/|/| 18/? [06:01<00:00, 7.16s/it]
100% 18/18 [05:02<00:00, 4.93s/it]
18/|/| 18/? [05:02<00:00, 4.93s/it]
Epoch 197 Loss => total: 7.1536 , g: 7.0901, l: 0.0175, t: 0.0373, p: 0.0087, theta: 69.248, theta_d: 7.636
100% 18/18 [06:03<00:00, 7.93s/it]
18/|/| 18/? [06:03<00:00, 7.93s/it]
100% 18/18 [05:02<00:00, 5.14s/it]
18/|/| 18/? [05:02<00:00, 5.14s/it]
Epoch 198 Loss => total: 7.0925 , g: 7.0294, l: 0.0173, t: 0.0370, p: 0.0087, theta: 69.513, theta_d: 22.809
100% 18/18 [05:57<00:00, 5.43s/it]
18/|/| 18/? [05:56<00:00, 5.43s/it]
100% 18/18 [05:01<00:00, 4.78s/it]
18/|/| 18/? [05:01<00:00, 4.78s/it]
Epoch 199 Loss => total: 7.0431 , g: 6.9802, l: 0.0172, t: 0.0368, p: 0.0090, theta: 68.984, theta_d: 352.480
100% 18/18 [06:03<00:00, 6.42s/it]
18/|/| 18/? [06:03<00:00, 6.42s/it]
100% 18/18 [05:04<00:00, 5.42s/it]
18/|/| 18/? [05:04<00:00, 5.42s/it]
Epoch 200 Loss => total: 6.8152 , g: 6.7533, l: 0.0169, t: 0.0360, p: 0.0090, theta: 68.342, theta_d: 315.731
100% 18/18 [06:02<00:00, 6.15s/it]
18/|/| 18/? [06:02<00:00, 6.15s/it]
100% 18/18 [05:02<00:00, 4.88s/it]
18/|/| 18/? [05:02<00:00, 4.88s/it]
Epoch 201 Loss => total: 6.7379 , g: 6.6788, l: 0.0168, t: 0.0357, p: 0.0066, theta: 67.589, theta_d: 272.590
100% 18/18 [05:54<00:00, 5.23s/it]
18/|/| 18/? [05:54<00:00, 5.23s/it]
100% 18/18 [05:02<00:00, 5.15s/it]
18/|/| 18/? [05:02<00:00, 5.15s/it]
Epoch 202 Loss => total: 6.6657 , g: 6.6034, l: 0.0167, t: 0.0354, p: 0.0103, theta: 67.834, theta_d: 286.621
100% 18/18 [06:09<00:00, 6.78s/it]
18/|/| 18/? [06:09<00:00, 6.78s/it]
100% 18/18 [05:03<00:00, 16.89s/it]
18/|/| 18/? [05:03<00:00, 16.89s/it]
Epoch 203 Loss => total: 6.6165 , g: 6.5579, l: 0.0165, t: 0.0351, p: 0.0069, theta: 67.711, theta_d: 279.558
100% 18/18 [06:03<00:00, 6.05s/it]
18/|/| 18/? [06:03<00:00, 6.05s/it]
100% 18/18 [05:01<00:00, 4.39s/it]
18/|/| 18/? [05:01<00:00, 4.39s/it]
Epoch 204 Loss => total: 6.5255 , g: 6.4673, l: 0.0164, t: 0.0348, p: 0.0070, theta: 67.684, theta_d: 277.982
100% 18/18 [06:03<00:00, 5.83s/it]
18/|/| 18/? [06:03<00:00, 5.83s/it]
100% 18/18 [05:03<00:00, 4.73s/it]
18/|/| 18/? [05:03<00:00, 4.73s/it]
Epoch 205 Loss => total: 27.6863 , g: 6.3994, l: 0.0163, t: 0.0345, p: 21.2361, theta: 66.884, theta_d: 232.152
100% 18/18 [06:04<00:00, 7.25s/it]
18/|/| 18/? [06:04<00:00, 7.25s/it]
100% 18/18 [05:03<00:00, 5.46s/it]
18/|/| 18/? [05:03<00:00, 5.46s/it]
Epoch 206 Loss => total: 27.6096 , g: 6.3268, l: 0.0162, t: 0.0342, p: 21.2324, theta: 65.887, theta_d: 175.036
100% 18/18 [06:02<00:00, 5.89s/it]
18/|/| 18/? [06:02<00:00, 5.89s/it]
100% 18/18 [05:03<00:00, 5.48s/it]
18/|/| 18/? [05:03<00:00, 5.48s/it]
Epoch 207 Loss => total: 27.5548 , g: 6.2598, l: 0.0161, t: 0.0339, p: 21.2450, theta: 66.017, theta_d: 182.509
100% 18/18 [05:58<00:00, 5.91s/it]
18/|/| 18/? [05:58<00:00, 5.90s/it]
100% 18/18 [05:04<00:00, 4.97s/it]
18/|/| 18/? [05:04<00:00, 4.97s/it]
Epoch 208 Loss => total: 27.4510 , g: 6.1422, l: 0.0159, t: 0.0335, p: 21.2594, theta: 66.017, theta_d: 182.487
100% 18/18 [05:55<00:00, 5.41s/it]
18/|/| 18/? [05:55<00:00, 5.41s/it]
100% 18/18 [05:07<00:00, 5.72s/it]
18/|/| 18/? [05:07<00:00, 5.72s/it]
Epoch 209 Loss => total: 27.3511 , g: 6.0441, l: 0.0157, t: 0.0331, p: 21.2582, theta: 65.822, theta_d: 171.296
100% 18/18 [05:59<00:00, 5.30s/it]
18/|/| 18/? [05:59<00:00, 5.30s/it]
100% 18/18 [05:02<00:00, 4.84s/it]
18/|/| 18/? [05:02<00:00, 4.84s/it]
Epoch 210 Loss => total: 27.2762 , g: 5.9723, l: 0.0156, t: 0.0328, p: 21.2555, theta: 66.026, theta_d: 183.025
100% 18/18 [06:01<00:00, 20.10s/it]
18/|/| 18/? [06:01<00:00, 20.10s/it]
100% 18/18 [05:06<00:00, 5.18s/it]
18/|/| 18/? [05:06<00:00, 5.18s/it]
Epoch 211 Loss => total: 27.1595 , g: 5.8553, l: 0.0154, t: 0.0324, p: 21.2564, theta: 66.065, theta_d: 185.247
100% 18/18 [06:01<00:00, 6.42s/it]
18/|/| 18/? [06:01<00:00, 6.42s/it]
100% 18/18 [05:02<00:00, 4.99s/it]
18/|/| 18/? [05:02<00:00, 4.99s/it]
Epoch 212 Loss => total: 27.0182 , g: 5.7160, l: 0.0152, t: 0.0319, p: 21.2550, theta: 65.931, theta_d: 177.583
100% 18/18 [06:00<00:00, 5.81s/it]
18/|/| 18/? [06:00<00:00, 5.81s/it]
100% 18/18 [05:03<00:00, 5.24s/it]
18/|/| 18/? [05:03<00:00, 5.24s/it]
Epoch 213 Loss => total: 26.9682 , g: 5.6686, l: 0.0151, t: 0.0317, p: 21.2529, theta: 66.002, theta_d: 181.633
100% 18/18 [06:00<00:00, 6.24s/it]
18/|/| 18/? [06:00<00:00, 6.24s/it]
100% 18/18 [05:02<00:00, 4.75s/it]
18/|/| 18/? [05:02<00:00, 4.75s/it]
Epoch 214 Loss => total: 26.8748 , g: 5.5786, l: 0.0150, t: 0.0313, p: 21.2499, theta: 65.980, theta_d: 180.352
100% 18/18 [06:06<00:00, 7.45s/it]
18/|/| 18/? [06:06<00:00, 7.45s/it]
100% 18/18 [05:03<00:00, 5.01s/it]
18/|/| 18/? [05:03<00:00, 5.01s/it]
Epoch 215 Loss => total: 26.8083 , g: 5.5175, l: 0.0149, t: 0.0310, p: 21.2448, theta: 65.972, theta_d: 179.925
100% 18/18 [05:58<00:00, 5.67s/it]
18/|/| 18/? [05:58<00:00, 5.67s/it]
100% 18/18 [05:03<00:00, 4.99s/it]
18/|/| 18/? [05:03<00:00, 4.99s/it]
Epoch 216 Loss => total: 26.7451 , g: 5.4616, l: 0.0148, t: 0.0308, p: 21.2379, theta: 65.973, theta_d: 180.003
100% 18/18 [06:05<00:00, 6.86s/it]
18/|/| 18/? [06:05<00:00, 6.86s/it]
100% 18/18 [05:02<00:00, 16.81s/it]
18/|/| 18/? [05:02<00:00, 16.81s/it]
Epoch 217 Loss => total: 26.6810 , g: 5.4067, l: 0.0147, t: 0.0305, p: 21.2291, theta: 65.973, theta_d: 180.001
100% 18/18 [05:57<00:00, 5.41s/it]
18/|/| 18/? [05:57<00:00, 5.41s/it]
100% 18/18 [05:04<00:00, 4.70s/it]
18/|/| 18/? [05:04<00:00, 4.70s/it]
Epoch 218 Loss => total: 26.6096 , g: 5.3469, l: 0.0146, t: 0.0303, p: 21.2178, theta: 65.973, theta_d: 180.001
100% 18/18 [06:05<00:00, 7.03s/it]
18/|/| 18/? [06:05<00:00, 7.03s/it]
100% 18/18 [05:03<00:00, 4.95s/it]
18/|/| 18/? [05:03<00:00, 4.95s/it]
Epoch 219 Loss => total: 26.5329 , g: 5.2846, l: 0.0145, t: 0.0301, p: 21.2038, theta: 65.973, theta_d: 180.001
100% 18/18 [06:04<00:00, 7.10s/it]
18/|/| 18/? [06:04<00:00, 7.10s/it]
100% 18/18 [05:03<00:00, 4.92s/it]
18/|/| 18/? [05:03<00:00, 4.92s/it]
Epoch 220 Loss => total: 26.4672 , g: 5.2366, l: 0.0143, t: 0.0298, p: 21.1865, theta: 65.973, theta_d: 180.001
100% 18/18 [06:07<00:00, 8.24s/it]
18/|/| 18/? [06:07<00:00, 8.24s/it]
100% 18/18 [05:02<00:00, 16.82s/it]
18/|/| 18/? [05:02<00:00, 16.81s/it]
Epoch 221 Loss => total: 26.3928 , g: 5.1839, l: 0.0143, t: 0.0296, p: 21.1651, theta: 65.973, theta_d: 180.002
100% 18/18 [06:00<00:00, 5.94s/it]
18/|/| 18/? [06:00<00:00, 5.94s/it]
100% 18/18 [05:08<00:00, 5.72s/it]
18/|/| 18/? [05:08<00:00, 5.72s/it]
Epoch 222 Loss => total: 26.3001 , g: 5.1198, l: 0.0142, t: 0.0293, p: 21.1368, theta: 65.973, theta_d: 180.002
100% 18/18 [06:02<00:00, 6.64s/it]
18/|/| 18/? [06:02<00:00, 6.64s/it]
100% 18/18 [05:06<00:00, 5.48s/it]
18/|/| 18/? [05:06<00:00, 5.48s/it]
Epoch 223 Loss => total: 26.1923 , g: 5.0758, l: 0.0141, t: 0.0291, p: 21.0734, theta: 65.973, theta_d: 180.003
100% 18/18 [06:04<00:00, 6.79s/it]
18/|/| 18/? [06:04<00:00, 6.79s/it]
100% 18/18 [05:03<00:00, 5.92s/it]
18/|/| 18/? [05:03<00:00, 5.92s/it]
Epoch 224 Loss => total: 6.3092 , g: 4.9948, l: 0.0139, t: 0.0287, p: 1.2718, theta: 65.968, theta_d: 179.671
100% 18/18 [05:59<00:00, 6.42s/it]
18/|/| 18/? [05:59<00:00, 6.42s/it]
100% 18/18 [05:02<00:00, 4.87s/it]
18/|/| 18/? [05:02<00:00, 4.87s/it]
Epoch 225 Loss => total: 6.2359 , g: 4.9393, l: 0.0137, t: 0.0285, p: 1.2544, theta: 65.963, theta_d: 179.395
100% 18/18 [06:03<00:00, 5.98s/it]
18/|/| 18/? [06:03<00:00, 5.98s/it]
100% 18/18 [05:04<00:00, 5.20s/it]
18/|/| 18/? [05:04<00:00, 5.20s/it]
Epoch 226 Loss => total: 6.1882 , g: 4.8934, l: 0.0136, t: 0.0283, p: 1.2529, theta: 65.959, theta_d: 179.186
100% 18/18 [06:01<00:00, 6.01s/it]
18/|/| 18/? [06:01<00:00, 6.01s/it]
100% 18/18 [05:06<00:00, 5.76s/it]
18/|/| 18/? [05:06<00:00, 5.76s/it]
Epoch 227 Loss => total: 6.1360 , g: 4.8417, l: 0.0135, t: 0.0280, p: 1.2527, theta: 65.958, theta_d: 179.123
100% 18/18 [06:01<00:00, 5.93s/it]
18/|/| 18/? [06:01<00:00, 5.93s/it]
100% 18/18 [05:04<00:00, 5.07s/it]
18/|/| 18/? [05:04<00:00, 5.07s/it]
Epoch 228 Loss => total: 6.0753 , g: 4.7812, l: 0.0134, t: 0.0278, p: 1.2529, theta: 65.958, theta_d: 179.109
100% 18/18 [05:58<00:00, 5.90s/it]
18/|/| 18/? [05:58<00:00, 5.90s/it]
100% 18/18 [05:03<00:00, 4.70s/it]
18/|/| 18/? [05:03<00:00, 4.70s/it]
Epoch 229 Loss => total: 6.0247 , g: 4.7309, l: 0.0132, t: 0.0275, p: 1.2531, theta: 65.958, theta_d: 179.103
100% 18/18 [06:00<00:00, 5.88s/it]
18/|/| 18/? [06:00<00:00, 5.88s/it]
100% 18/18 [05:02<00:00, 4.47s/it]
18/|/| 18/? [05:02<00:00, 4.47s/it]
Epoch 230 Loss => total: 5.9843 , g: 4.6906, l: 0.0131, t: 0.0273, p: 1.2533, theta: 65.958, theta_d: 179.096
100% 18/18 [06:01<00:00, 5.90s/it]
18/|/| 18/? [06:01<00:00, 5.90s/it]
100% 18/18 [05:06<00:00, 4.75s/it]
18/|/| 18/? [05:06<00:00, 4.75s/it]
Epoch 231 Loss => total: 5.9227 , g: 4.6291, l: 0.0130, t: 0.0270, p: 1.2535, theta: 65.958, theta_d: 179.088
100% 18/18 [06:02<00:00, 5.97s/it]
18/|/| 18/? [06:02<00:00, 5.97s/it]
100% 18/18 [05:03<00:00, 3.83s/it]
18/|/| 18/? [05:03<00:00, 3.83s/it]
Epoch 232 Loss => total: 5.8643 , g: 4.5710, l: 0.0129, t: 0.0268, p: 1.2537, theta: 65.957, theta_d: 179.077
100% 18/18 [05:58<00:00, 6.36s/it]
18/|/| 18/? [05:58<00:00, 6.36s/it]
100% 18/18 [05:03<00:00, 16.84s/it]
18/|/| 18/? [05:03<00:00, 16.84s/it]
Epoch 233 Loss => total: 5.8152 , g: 4.5220, l: 0.0128, t: 0.0266, p: 1.2538, theta: 65.957, theta_d: 179.064
100% 18/18 [06:01<00:00, 6.55s/it]
18/|/| 18/? [06:01<00:00, 6.55s/it]
100% 18/18 [05:03<00:00, 7.40s/it]
18/|/| 18/? [05:02<00:00, 7.40s/it]
Epoch 234 Loss => total: 5.7608 , g: 4.4678, l: 0.0127, t: 0.0263, p: 1.2539, theta: 65.957, theta_d: 179.049
100% 18/18 [06:00<00:00, 5.85s/it]
18/|/| 18/? [06:00<00:00, 5.85s/it]
100% 18/18 [05:02<00:00, 4.88s/it]
18/|/| 18/? [05:02<00:00, 4.88s/it]
Epoch 235 Loss => total: 5.7168 , g: 4.4241, l: 0.0126, t: 0.0261, p: 1.2541, theta: 65.957, theta_d: 179.030
100% 18/18 [05:59<00:00, 5.76s/it]
18/|/| 18/? [05:59<00:00, 5.76s/it]
100% 18/18 [05:02<00:00, 4.77s/it]
18/|/| 18/? [05:02<00:00, 4.77s/it]
Epoch 236 Loss => total: 5.6649 , g: 4.3724, l: 0.0125, t: 0.0258, p: 1.2542, theta: 65.956, theta_d: 179.008
100% 18/18 [06:01<00:00, 8.14s/it]
18/|/| 18/? [06:01<00:00, 8.14s/it]
100% 18/18 [05:07<00:00, 5.65s/it]
18/|/| 18/? [05:07<00:00, 5.65s/it]
Epoch 237 Loss => total: 5.6220 , g: 4.3298, l: 0.0123, t: 0.0256, p: 1.2543, theta: 65.956, theta_d: 178.981
100% 18/18 [05:59<00:00, 5.49s/it]
18/|/| 18/? [05:59<00:00, 5.49s/it]
100% 18/18 [05:01<00:00, 4.78s/it]
18/|/| 18/? [05:01<00:00, 4.78s/it]
Epoch 238 Loss => total: 5.5515 , g: 4.2595, l: 0.0122, t: 0.0254, p: 1.2544, theta: 65.955, theta_d: 178.950
100% 18/18 [06:03<00:00, 6.19s/it]
18/|/| 18/? [06:03<00:00, 6.19s/it]
100% 18/18 [05:03<00:00, 5.08s/it]
18/|/| 18/? [05:02<00:00, 5.08s/it]
Epoch 239 Loss => total: 5.5092 , g: 4.2174, l: 0.0121, t: 0.0251, p: 1.2545, theta: 65.954, theta_d: 178.913
100% 18/18 [06:01<00:00, 5.61s/it]
18/|/| 18/? [06:01<00:00, 5.61s/it]
100% 18/18 [05:02<00:00, 4.47s/it]
18/|/| 18/? [05:02<00:00, 4.47s/it]
Epoch 240 Loss => total: 5.4621 , g: 4.1706, l: 0.0120, t: 0.0249, p: 1.2546, theta: 65.954, theta_d: 178.870
100% 18/18 [06:03<00:00, 6.17s/it]
18/|/| 18/? [06:03<00:00, 6.17s/it]
100% 18/18 [05:01<00:00, 4.59s/it]
18/|/| 18/? [05:01<00:00, 4.59s/it]
Epoch 241 Loss => total: 5.4057 , g: 4.1144, l: 0.0119, t: 0.0246, p: 1.2547, theta: 65.953, theta_d: 178.820
100% 18/18 [06:02<00:00, 5.87s/it]
18/|/| 18/? [06:01<00:00, 5.87s/it]
100% 18/18 [05:04<00:00, 5.40s/it]
18/|/| 18/? [05:04<00:00, 5.40s/it]
Epoch 242 Loss => total: 5.3337 , g: 4.0427, l: 0.0118, t: 0.0244, p: 1.2548, theta: 65.952, theta_d: 178.761
100% 18/18 [05:59<00:00, 5.87s/it]
18/|/| 18/? [05:59<00:00, 5.87s/it]
100% 18/18 [05:02<00:00, 4.61s/it]
18/|/| 18/? [05:02<00:00, 4.61s/it]
Epoch 243 Loss => total: 5.2813 , g: 3.9906, l: 0.0117, t: 0.0242, p: 1.2549, theta: 65.951, theta_d: 178.692
100% 18/18 [06:00<00:00, 5.66s/it]
18/|/| 18/? [06:00<00:00, 5.66s/it]
100% 18/18 [05:02<00:00, 4.76s/it]
18/|/| 18/? [05:02<00:00, 4.76s/it]
Epoch 244 Loss => total: 5.2279 , g: 3.9374, l: 0.0116, t: 0.0239, p: 1.2550, theta: 65.949, theta_d: 178.612
100% 18/18 [06:02<00:00, 7.54s/it]
18/|/| 18/? [06:02<00:00, 7.54s/it]
100% 18/18 [05:04<00:00, 5.00s/it]
18/|/| 18/? [05:04<00:00, 5.00s/it]
Epoch 245 Loss => total: 5.1767 , g: 3.8865, l: 0.0115, t: 0.0237, p: 1.2550, theta: 65.948, theta_d: 178.518
100% 18/18 [06:00<00:00, 5.93s/it]
18/|/| 18/? [06:00<00:00, 5.93s/it]
100% 18/18 [05:01<00:00, 4.53s/it]
18/|/| 18/? [05:01<00:00, 4.53s/it]
Epoch 246 Loss => total: 5.1227 , g: 3.8328, l: 0.0113, t: 0.0234, p: 1.2551, theta: 65.946, theta_d: 178.407
100% 18/18 [06:00<00:00, 6.06s/it]
18/|/| 18/? [06:00<00:00, 6.06s/it]
100% 18/18 [05:04<00:00, 5.30s/it]
18/|/| 18/? [05:04<00:00, 5.30s/it]
Epoch 247 Loss => total: 5.0669 , g: 3.7772, l: 0.0112, t: 0.0232, p: 1.2552, theta: 65.943, theta_d: 178.277
100% 18/18 [05:58<00:00, 5.95s/it]
18/|/| 18/? [05:58<00:00, 5.95s/it]
100% 18/18 [05:03<00:00, 4.81s/it]
18/|/| 18/? [05:03<00:00, 4.81s/it]
Epoch 248 Loss => total: 5.0247 , g: 3.7354, l: 0.0111, t: 0.0229, p: 1.2553, theta: 65.941, theta_d: 178.121
100% 18/18 [05:58<00:00, 6.04s/it]
18/|/| 18/? [05:58<00:00, 6.04s/it]
100% 18/18 [05:00<00:00, 16.69s/it]
18/|/| 18/? [05:00<00:00, 16.69s/it]
Epoch 249 Loss => total: 4.9728 , g: 3.6838, l: 0.0110, t: 0.0227, p: 1.2553, theta: 65.937, theta_d: 177.932
100% 18/18 [06:02<00:00, 6.16s/it]
18/|/| 18/? [06:02<00:00, 6.16s/it]
100% 18/18 [05:06<00:00, 5.02s/it]
18/|/| 18/? [05:06<00:00, 5.02s/it]
Epoch 250 Loss => total: 4.9172 , g: 3.6285, l: 0.0109, t: 0.0224, p: 1.2554, theta: 65.933, theta_d: 177.690
100% 18/18 [05:57<00:00, 5.35s/it]
18/|/| 18/? [05:57<00:00, 5.35s/it]
100% 18/18 [05:03<00:00, 5.22s/it]
18/|/| 18/? [05:03<00:00, 5.22s/it]
Epoch 251 Loss => total: 4.8522 , g: 3.5639, l: 0.0108, t: 0.0222, p: 1.2554, theta: 65.927, theta_d: 177.359
100% 18/18 [06:03<00:00, 6.58s/it]
18/|/| 18/? [06:03<00:00, 6.58s/it]
100% 18/18 [05:01<00:00, 4.44s/it]
18/|/| 18/? [05:01<00:00, 4.44s/it]
Epoch 252 Loss => total: 4.7828 , g: 3.4948, l: 0.0106, t: 0.0219, p: 1.2555, theta: 65.919, theta_d: 176.856
100% 18/18 [06:00<00:00, 5.68s/it]
18/|/| 18/? [06:00<00:00, 5.68s/it]
100% 18/18 [05:01<00:00, 4.54s/it]
18/|/| 18/? [05:01<00:00, 4.54s/it]
Epoch 253 Loss => total: 4.7099 , g: 3.4223, l: 0.0105, t: 0.0216, p: 1.2555, theta: 65.904, theta_d: 176.016
100% 18/18 [06:02<00:00, 5.74s/it]
18/|/| 18/? [06:02<00:00, 5.74s/it]
100% 18/18 [05:06<00:00, 5.02s/it]
18/|/| 18/? [05:06<00:00, 5.02s/it]
Epoch 254 Loss => total: 4.6567 , g: 3.3697, l: 0.0104, t: 0.0213, p: 1.2553, theta: 65.877, theta_d: 174.446
100% 18/18 [05:58<00:00, 5.46s/it]
18/|/| 18/? [05:58<00:00, 5.46s/it]
100% 18/18 [05:04<00:00, 5.22s/it]
18/|/| 18/? [05:04<00:00, 5.22s/it]
Epoch 255 Loss => total: 4.5493 , g: 3.2642, l: 0.0102, t: 0.0209, p: 1.2539, theta: 65.799, theta_d: 169.994
100% 18/18 [06:00<00:00, 6.27s/it]
18/|/| 18/? [06:00<00:00, 6.27s/it]
100% 18/18 [05:03<00:00, 4.97s/it]
18/|/| 18/? [05:03<00:00, 4.97s/it]
Epoch 256 Loss => total: 4.4174 , g: 3.1421, l: 0.0100, t: 0.0204, p: 1.2449, theta: 65.599, theta_d: 158.526
100% 18/18 [05:59<00:00, 5.68s/it]
18/|/| 18/? [05:59<00:00, 5.68s/it]
100% 18/18 [05:03<00:00, 4.93s/it]
18/|/| 18/? [05:03<00:00, 4.93s/it]
Epoch 257 Loss => total: 4.3456 , g: 3.0888, l: 0.0099, t: 0.0202, p: 1.2268, theta: 65.502, theta_d: 153.016
100% 18/18 [06:03<00:00, 5.97s/it]
18/|/| 18/? [06:03<00:00, 5.97s/it]
100% 18/18 [05:04<00:00, 5.20s/it]
18/|/| 18/? [05:03<00:00, 5.20s/it]
Epoch 258 Loss => total: 4.2657 , g: 3.0365, l: 0.0098, t: 0.0200, p: 1.1994, theta: 65.473, theta_d: 151.337
100% 18/18 [06:00<00:00, 6.28s/it]
18/|/| 18/? [06:00<00:00, 6.28s/it]
100% 18/18 [05:01<00:00, 16.77s/it]
18/|/| 18/? [05:01<00:00, 16.76s/it]
Epoch 259 Loss => total: 4.1961 , g: 2.9946, l: 0.0097, t: 0.0198, p: 1.1720, theta: 65.638, theta_d: 160.801
100% 18/18 [05:59<00:00, 6.50s/it]
18/|/| 18/? [05:59<00:00, 6.50s/it]
100% 18/18 [05:05<00:00, 5.16s/it]
18/|/| 18/? [05:05<00:00, 5.16s/it]
Epoch 260 Loss => total: 4.1193 , g: 2.9170, l: 0.0096, t: 0.0195, p: 1.1732, theta: 66.567, theta_d: 213.980
100% 18/18 [05:57<00:00, 6.11s/it]
18/|/| 18/? [05:57<00:00, 6.11s/it]
100% 18/18 [05:02<00:00, 4.77s/it]
18/|/| 18/? [05:02<00:00, 4.78s/it]
Epoch 261 Loss => total: 4.0925 , g: 2.9109, l: 0.0095, t: 0.0194, p: 1.1528, theta: 66.753, theta_d: 224.655
100% 18/18 [05:59<00:00, 6.00s/it]
18/|/| 18/? [05:59<00:00, 6.00s/it]
100% 18/18 [05:05<00:00, 4.27s/it]
18/|/| 18/? [05:05<00:00, 4.27s/it]
Epoch 262 Loss => total: 20.3420 , g: 2.9079, l: 0.0094, t: 0.0193, p: 17.4054, theta: 67.868, theta_d: 288.522
100% 18/18 [05:58<00:00, 5.86s/it]
18/|/| 18/? [05:58<00:00, 5.86s/it]
100% 18/18 [05:05<00:00, 5.05s/it]
18/|/| 18/? [05:05<00:00, 5.05s/it]
Epoch 263 Loss => total: 3.4741 , g: 2.8239, l: 0.0093, t: 0.0190, p: 0.6218, theta: 72.904, theta_d: 217.073
100% 18/18 [05:57<00:00, 5.61s/it]
18/|/| 18/? [05:57<00:00, 5.61s/it]
100% 18/18 [05:07<00:00, 5.91s/it]
18/|/| 18/? [05:07<00:00, 5.91s/it]
Epoch 264 Loss => total: 3.9756 , g: 2.8024, l: 0.0092, t: 0.0188, p: 1.1451, theta: 73.136, theta_d: 230.390
100% 18/18 [06:00<00:00, 6.20s/it]
18/|/| 18/? [06:00<00:00, 6.20s/it]
100% 18/18 [05:07<00:00, 5.25s/it]
18/|/| 18/? [05:07<00:00, 5.25s/it]
Epoch 265 Loss => total: 22.1892 , g: 2.7880, l: 0.0092, t: 0.0187, p: 19.3732, theta: 75.515, theta_d: 6.667
100% 18/18 [06:02<00:00, 6.32s/it]
18/|/| 18/? [06:02<00:00, 6.32s/it]
100% 18/18 [05:04<00:00, 5.14s/it]
18/|/| 18/? [05:04<00:00, 5.14s/it]
Epoch 266 Loss => total: 20.3272 , g: 2.7177, l: 0.0091, t: 0.0184, p: 17.5820, theta: 76.496, theta_d: 62.873
100% 18/18 [06:04<00:00, 6.39s/it]
18/|/| 18/? [06:04<00:00, 6.39s/it]
100% 18/18 [05:07<00:00, 5.60s/it]
18/|/| 18/? [05:07<00:00, 5.60s/it]
Epoch 267 Loss => total: 3.6946 , g: 2.6628, l: 0.0090, t: 0.0182, p: 1.0046, theta: 80.162, theta_d: 272.934
100% 18/18 [06:01<00:00, 5.49s/it]
18/|/| 18/? [06:01<00:00, 5.49s/it]
100% 18/18 [05:04<00:00, 4.85s/it]
18/|/| 18/? [05:04<00:00, 4.85s/it]
Epoch 268 Loss => total: 2.9120 , g: 2.6452, l: 0.0089, t: 0.0181, p: 0.2397, theta: 99.410, theta_d: 295.763
100% 18/18 [06:09<00:00, 6.40s/it]
18/|/| 18/? [06:09<00:00, 6.40s/it]
100% 18/18 [05:08<00:00, 5.05s/it]
18/|/| 18/? [05:08<00:00, 5.05s/it]
Epoch 269 Loss => total: 2.6404 , g: 2.5885, l: 0.0088, t: 0.0179, p: 0.0253, theta: 99.767, theta_d: 316.228
100% 18/18 [06:06<00:00, 5.83s/it]
18/|/| 18/? [06:06<00:00, 5.83s/it]
100% 18/18 [05:15<00:00, 6.03s/it]
18/|/| 18/? [05:15<00:00, 6.03s/it]
Epoch 270 Loss => total: 2.5896 , g: 2.5379, l: 0.0087, t: 0.0177, p: 0.0253, theta: 100.900, theta_d: 21.118
100% 18/18 [06:05<00:00, 5.76s/it]
18/|/| 18/? [06:05<00:00, 5.76s/it]
100% 18/18 [05:13<00:00, 5.98s/it]
18/|/| 18/? [05:13<00:00, 5.98s/it]
Epoch 271 Loss => total: 2.5654 , g: 2.5159, l: 0.0086, t: 0.0176, p: 0.0233, theta: 100.974, theta_d: 25.389
100% 18/18 [06:13<00:00, 6.63s/it]
18/|/| 18/? [06:13<00:00, 6.63s/it]
100% 18/18 [05:12<00:00, 5.38s/it]
18/|/| 18/? [05:12<00:00, 5.38s/it]
Epoch 272 Loss => total: 2.5507 , g: 2.5055, l: 0.0086, t: 0.0175, p: 0.0191, theta: 101.057, theta_d: 30.127
100% 18/18 [06:07<00:00, 5.64s/it]
18/|/| 18/? [06:07<00:00, 5.64s/it]
100% 18/18 [05:19<00:00, 5.55s/it]
18/|/| 18/? [05:19<00:00, 5.55s/it]
Epoch 273 Loss => total: 2.5360 , g: 2.4953, l: 0.0085, t: 0.0174, p: 0.0148, theta: 101.039, theta_d: 29.096
100% 18/18 [06:11<00:00, 5.69s/it]
18/|/| 18/? [06:11<00:00, 5.69s/it]
100% 18/18 [05:13<00:00, 5.39s/it]
18/|/| 18/? [05:13<00:00, 5.39s/it]
Epoch 274 Loss => total: 2.5126 , g: 2.4783, l: 0.0085, t: 0.0173, p: 0.0085, theta: 101.078, theta_d: 31.324
100% 18/18 [06:12<00:00, 5.68s/it]
18/|/| 18/? [06:12<00:00, 5.68s/it]
100% 18/18 [05:13<00:00, 5.05s/it]
18/|/| 18/? [05:13<00:00, 5.05s/it]
Epoch 275 Loss => total: 2.5143 , g: 2.4818, l: 0.0084, t: 0.0172, p: 0.0070, theta: 101.229, theta_d: 40.021
100% 18/18 [06:16<00:00, 5.29s/it]
18/|/| 18/? [06:16<00:00, 5.29s/it]
100% 18/18 [05:19<00:00, 4.41s/it]
18/|/| 18/? [05:19<00:00, 4.41s/it]
Epoch 276 Loss => total: 2.4965 , g: 2.4646, l: 0.0084, t: 0.0171, p: 0.0065, theta: 101.378, theta_d: 48.531
100% 18/18 [06:28<00:00, 6.53s/it]
18/|/| 18/? [06:28<00:00, 6.53s/it]
100% 18/18 [05:07<00:00, 4.48s/it]
18/|/| 18/? [05:07<00:00, 4.48s/it]
Epoch 277 Loss => total: 2.4815 , g: 2.4500, l: 0.0083, t: 0.0170, p: 0.0062, theta: 101.515, theta_d: 56.358
100% 18/18 [06:18<00:00, 21.05s/it]
18/|/| 18/? [06:18<00:00, 21.05s/it]
100% 18/18 [05:11<00:00, 4.96s/it]
18/|/| 18/? [05:11<00:00, 4.96s/it]
Epoch 278 Loss => total: 2.4683 , g: 2.4363, l: 0.0083, t: 0.0169, p: 0.0069, theta: 101.639, theta_d: 63.513
100% 18/18 [06:17<00:00, 5.60s/it]
18/|/| 18/? [06:17<00:00, 5.60s/it]
100% 18/18 [05:09<00:00, 6.28s/it]
18/|/| 18/? [05:09<00:00, 6.28s/it]
Epoch 279 Loss => total: 2.4588 , g: 2.4241, l: 0.0082, t: 0.0168, p: 0.0097, theta: 101.754, theta_d: 70.098
100% 18/18 [06:22<00:00, 6.29s/it]
18/|/| 18/? [06:22<00:00, 6.29s/it]
100% 18/18 [05:16<00:00, 5.45s/it]
18/|/| 18/? [05:16<00:00, 5.45s/it]
Epoch 280 Loss => total: 2.4599 , g: 2.4189, l: 0.0082, t: 0.0167, p: 0.0162, theta: 101.861, theta_d: 76.224
100% 18/18 [06:18<00:00, 6.24s/it]
18/|/| 18/? [06:18<00:00, 6.24s/it]
100% 18/18 [05:09<00:00, 4.65s/it]
18/|/| 18/? [05:09<00:00, 4.65s/it]
Epoch 281 Loss => total: 2.4620 , g: 2.4071, l: 0.0081, t: 0.0166, p: 0.0302, theta: 101.962, theta_d: 81.967
100% 18/18 [06:17<00:00, 6.37s/it]
18/|/| 18/? [06:17<00:00, 6.37s/it]
100% 18/18 [05:09<00:00, 4.97s/it]
18/|/| 18/? [05:09<00:00, 4.97s/it]
Epoch 282 Loss => total: 2.5046 , g: 2.3963, l: 0.0081, t: 0.0165, p: 0.0837, theta: 102.056, theta_d: 87.377
100% 18/18 [06:25<00:00, 6.26s/it]
18/|/| 18/? [06:25<00:00, 6.26s/it]
100% 18/18 [05:08<00:00, 17.15s/it]
18/|/| 18/? [05:08<00:00, 17.15s/it]
Epoch 283 Loss => total: 3.0819 , g: 2.3876, l: 0.0080, t: 0.0164, p: 0.6698, theta: 102.145, theta_d: 92.491
100% 18/18 [06:29<00:00, 6.69s/it]
18/|/| 18/? [06:29<00:00, 6.69s/it]
100% 18/18 [05:08<00:00, 4.85s/it]
18/|/| 18/? [05:08<00:00, 4.85s/it]
Epoch 284 Loss => total: 3.4794 , g: 2.3708, l: 0.0080, t: 0.0163, p: 1.0843, theta: 102.230, theta_d: 97.343
100% 18/18 [06:31<00:00, 6.09s/it]
18/|/| 18/? [06:31<00:00, 6.09s/it]
100% 18/18 [05:09<00:00, 5.03s/it]
18/|/| 18/? [05:09<00:00, 5.03s/it]
Epoch 285 Loss => total: 2.4844 , g: 2.3536, l: 0.0079, t: 0.0162, p: 0.1067, theta: 102.306, theta_d: 101.713
100% 18/18 [06:40<00:00, 6.52s/it]
18/|/| 18/? [06:40<00:00, 6.52s/it]
100% 18/18 [05:08<00:00, 5.05s/it]
18/|/| 18/? [05:08<00:00, 5.05s/it]
Epoch 286 Loss => total: 6.9039 , g: 2.3182, l: 0.0079, t: 0.0161, p: 4.5618, theta: 112.633, theta_d: 333.402
100% 18/18 [06:34<00:00, 6.80s/it]
18/|/| 18/? [06:34<00:00, 6.80s/it]
100% 18/18 [05:07<00:00, 4.48s/it]
18/|/| 18/? [05:07<00:00, 4.48s/it]
Epoch 287 Loss => total: 2.3822 , g: 2.2682, l: 0.0078, t: 0.0158, p: 0.0904, theta: 114.612, theta_d: 86.797
100% 18/18 [06:51<00:00, 9.36s/it]
18/|/| 18/? [06:51<00:00, 9.36s/it]
100% 18/18 [05:12<00:00, 5.14s/it]
18/|/| 18/? [05:12<00:00, 5.14s/it]
Epoch 288 Loss => total: 2.7863 , g: 2.2607, l: 0.0077, t: 0.0158, p: 0.5021, theta: 114.702, theta_d: 91.951
100% 18/18 [06:48<00:00, 6.54s/it]
18/|/| 18/? [06:48<00:00, 6.54s/it]
100% 18/18 [05:09<00:00, 5.09s/it]
18/|/| 18/? [05:09<00:00, 5.09s/it]
Epoch 289 Loss => total: 3.4591 , g: 2.2465, l: 0.0077, t: 0.0157, p: 1.1892, theta: 114.787, theta_d: 96.834
100% 18/18 [06:36<00:00, 5.81s/it]
18/|/| 18/? [06:36<00:00, 5.81s/it]
100% 18/18 [05:09<00:00, 4.83s/it]
18/|/| 18/? [05:09<00:00, 4.83s/it]
Epoch 290 Loss => total: 2.4924 , g: 2.2287, l: 0.0076, t: 0.0155, p: 0.2406, theta: 114.866, theta_d: 101.361
100% 18/18 [06:23<00:00, 6.55s/it]
18/|/| 18/? [06:23<00:00, 6.55s/it]
100% 18/18 [05:08<00:00, 4.99s/it]
18/|/| 18/? [05:08<00:00, 4.99s/it]
Epoch 291 Loss => total: 2.3884 , g: 2.2236, l: 0.0076, t: 0.0154, p: 0.1418, theta: 114.926, theta_d: 104.790
100% 18/18 [06:35<00:00, 6.10s/it]
18/|/| 18/? [06:35<00:00, 6.10s/it]
100% 18/18 [05:10<00:00, 4.82s/it]
18/|/| 18/? [05:10<00:00, 4.82s/it]
Epoch 292 Loss => total: 2.4277 , g: 2.1921, l: 0.0075, t: 0.0153, p: 0.2128, theta: 121.198, theta_d: 104.157
100% 18/18 [06:27<00:00, 6.71s/it]
18/|/| 18/? [06:27<00:00, 6.71s/it]
100% 18/18 [05:09<00:00, 4.42s/it]
18/|/| 18/? [05:09<00:00, 4.42s/it]
Epoch 293 Loss => total: 2.4077 , g: 2.1703, l: 0.0075, t: 0.0152, p: 0.2146, theta: 123.163, theta_d: 216.709
100% 18/18 [06:29<00:00, 5.81s/it]
18/|/| 18/? [06:29<00:00, 5.81s/it]
100% 18/18 [05:12<00:00, 5.05s/it]
18/|/| 18/? [05:12<00:00, 5.05s/it]
Epoch 294 Loss => total: 2.5775 , g: 2.1676, l: 0.0074, t: 0.0152, p: 0.3872, theta: 133.562, theta_d: 92.537
100% 18/18 [06:34<00:00, 6.59s/it]
18/|/| 18/? [06:34<00:00, 6.59s/it]
100% 18/18 [05:15<00:00, 5.39s/it]
18/|/| 18/? [05:15<00:00, 5.39s/it]
Epoch 295 Loss => total: 5.2324 , g: 2.1347, l: 0.0074, t: 0.0151, p: 3.0752, theta: 138.155, theta_d: 355.687
100% 18/18 [06:57<00:00, 7.62s/it]
18/|/| 18/? [06:57<00:00, 7.62s/it]
100% 18/18 [05:15<00:00, 5.00s/it]
18/|/| 18/? [05:15<00:00, 5.00s/it]
Epoch 296 Loss => total: 2.1447 , g: 2.1210, l: 0.0073, t: 0.0150, p: 0.0014, theta: 149.982, theta_d: 313.363
100% 18/18 [06:57<00:00, 5.54s/it]
18/|/| 18/? [06:57<00:00, 5.54s/it]
100% 18/18 [05:10<00:00, 5.04s/it]
18/|/| 18/? [05:10<00:00, 5.04s/it]
Epoch 297 Loss => total: 2.1413 , g: 2.1178, l: 0.0073, t: 0.0149, p: 0.0013, theta: 150.003, theta_d: 314.543
100% 18/18 [06:43<00:00, 4.72s/it]
18/|/| 18/? [06:43<00:00, 4.72s/it]
100% 18/18 [05:14<00:00, 5.18s/it]
18/|/| 18/? [05:14<00:00, 5.18s/it]
Epoch 298 Loss => total: 2.1352 , g: 2.1116, l: 0.0073, t: 0.0149, p: 0.0014, theta: 150.021, theta_d: 315.570
100% 18/18 [06:42<00:00, 5.91s/it]
18/|/| 18/? [06:42<00:00, 5.91s/it]
100% 18/18 [05:10<00:00, 5.16s/it]
18/|/| 18/? [05:10<00:00, 5.16s/it]
Epoch 299 Loss => total: 2.1234 , g: 2.0997, l: 0.0073, t: 0.0148, p: 0.0016, theta: 150.037, theta_d: 316.492
100% 18/18 [06:34<00:00, 6.65s/it]
18/|/| 18/? [06:34<00:00, 6.65s/it]
100% 18/18 [05:12<00:00, 5.40s/it]
18/|/| 18/? [05:12<00:00, 5.40s/it]
Epoch 300 Loss => total: 2.1161 , g: 2.0924, l: 0.0072, t: 0.0148, p: 0.0017, theta: 150.052, theta_d: 317.344
100% 18/18 [06:27<00:00, 5.85s/it]
18/|/| 18/? [06:27<00:00, 5.85s/it]
100% 18/18 [05:11<00:00, 4.40s/it]
18/|/| 18/? [05:11<00:00, 4.40s/it]
Epoch 301 Loss => total: 2.1080 , g: 2.0842, l: 0.0072, t: 0.0147, p: 0.0019, theta: 150.065, theta_d: 318.115
100% 18/18 [06:31<00:00, 6.31s/it]
18/|/| 18/? [06:30<00:00, 6.31s/it]
100% 18/18 [05:13<00:00, 5.51s/it]
18/|/| 18/? [05:13<00:00, 5.51s/it]
Epoch 302 Loss => total: 2.1014 , g: 2.0776, l: 0.0072, t: 0.0147, p: 0.0020, theta: 150.088, theta_d: 319.397
100% 18/18 [06:31<00:00, 6.48s/it]
18/|/| 18/? [06:31<00:00, 6.48s/it]
100% 18/18 [05:09<00:00, 4.72s/it]
18/|/| 18/? [05:09<00:00, 4.72s/it]
Epoch 303 Loss => total: 2.0939 , g: 2.0700, l: 0.0072, t: 0.0146, p: 0.0021, theta: 150.110, theta_d: 320.654
100% 18/18 [06:21<00:00, 6.51s/it]
18/|/| 18/? [06:20<00:00, 6.51s/it]
100% 18/18 [05:08<00:00, 5.04s/it]
18/|/| 18/? [05:08<00:00, 5.04s/it]
Epoch 304 Loss => total: 2.0957 , g: 2.0718, l: 0.0071, t: 0.0146, p: 0.0022, theta: 150.131, theta_d: 321.892
100% 18/18 [06:29<00:00, 6.31s/it]
18/|/| 18/? [06:29<00:00, 6.31s/it]
100% 18/18 [05:08<00:00, 6.43s/it]
18/|/| 18/? [05:08<00:00, 6.43s/it]
Epoch 305 Loss => total: 2.0802 , g: 2.0563, l: 0.0071, t: 0.0145, p: 0.0023, theta: 150.153, theta_d: 323.118
100% 18/18 [06:39<00:00, 6.46s/it]
18/|/| 18/? [06:39<00:00, 6.46s/it]
100% 18/18 [05:12<00:00, 6.35s/it]
18/|/| 18/? [05:12<00:00, 6.35s/it]
Epoch 306 Loss => total: 2.0712 , g: 2.0474, l: 0.0071, t: 0.0144, p: 0.0024, theta: 150.174, theta_d: 324.339
100% 18/18 [06:47<00:00, 7.31s/it]
18/|/| 18/? [06:47<00:00, 7.31s/it]
100% 18/18 [05:16<00:00, 17.60s/it]
18/|/| 18/? [05:16<00:00, 17.60s/it]
Epoch 307 Loss => total: 2.0648 , g: 2.0409, l: 0.0070, t: 0.0144, p: 0.0024, theta: 150.195, theta_d: 325.560
100% 18/18 [06:37<00:00, 5.94s/it]
18/|/| 18/? [06:37<00:00, 5.94s/it]
100% 18/18 [05:12<00:00, 4.39s/it]
18/|/| 18/? [05:12<00:00, 4.39s/it]
Epoch 308 Loss => total: 2.0577 , g: 2.0339, l: 0.0070, t: 0.0143, p: 0.0024, theta: 150.217, theta_d: 326.780
100% 18/18 [06:25<00:00, 5.55s/it]
18/|/| 18/? [06:25<00:00, 5.55s/it]
100% 18/18 [05:15<00:00, 6.66s/it]
18/|/| 18/? [05:15<00:00, 6.66s/it]
Epoch 309 Loss => total: 2.0529 , g: 2.0292, l: 0.0070, t: 0.0143, p: 0.0024, theta: 150.238, theta_d: 327.998
100% 18/18 [06:26<00:00, 5.35s/it]
18/|/| 18/? [06:26<00:00, 5.35s/it]
100% 18/18 [05:14<00:00, 4.92s/it]
18/|/| 18/? [05:14<00:00, 4.92s/it]
Epoch 310 Loss => total: 2.0450 , g: 2.0214, l: 0.0070, t: 0.0142, p: 0.0024, theta: 150.259, theta_d: 329.210
100% 18/18 [06:35<00:00, 5.90s/it]
18/|/| 18/? [06:35<00:00, 5.90s/it]
100% 18/18 [05:22<00:00, 5.34s/it]
18/|/| 18/? [05:22<00:00, 5.34s/it]
Epoch 311 Loss => total: 2.0374 , g: 2.0139, l: 0.0069, t: 0.0142, p: 0.0023, theta: 150.280, theta_d: 330.409
100% 18/18 [06:40<00:00, 6.10s/it]
18/|/| 18/? [06:40<00:00, 6.10s/it]
100% 18/18 [05:18<00:00, 4.77s/it]
18/|/| 18/? [05:18<00:00, 4.77s/it]
Epoch 312 Loss => total: 2.0349 , g: 2.0117, l: 0.0069, t: 0.0141, p: 0.0022, theta: 150.301, theta_d: 331.589
100% 18/18 [06:30<00:00, 6.31s/it]
18/|/| 18/? [06:30<00:00, 6.31s/it]
100% 18/18 [05:14<00:00, 5.00s/it]
18/|/| 18/? [05:14<00:00, 5.00s/it]
Epoch 313 Loss => total: 2.0326 , g: 2.0097, l: 0.0069, t: 0.0141, p: 0.0021, theta: 150.321, theta_d: 332.744
100% 18/18 [06:29<00:00, 6.07s/it]
18/|/| 18/? [06:29<00:00, 6.07s/it]
100% 18/18 [05:13<00:00, 4.62s/it]
18/|/| 18/? [05:13<00:00, 4.62s/it]
Epoch 314 Loss => total: 2.0200 , g: 1.9973, l: 0.0069, t: 0.0140, p: 0.0019, theta: 150.340, theta_d: 333.867
100% 18/18 [06:31<00:00, 5.84s/it]
18/|/| 18/? [06:31<00:00, 5.84s/it]
100% 18/18 [05:11<00:00, 5.04s/it]
18/|/| 18/? [05:11<00:00, 5.04s/it]
Epoch 315 Loss => total: 2.0113 , g: 1.9888, l: 0.0068, t: 0.0139, p: 0.0017, theta: 150.359, theta_d: 334.953
100% 18/18 [06:34<00:00, 6.24s/it]
18/|/| 18/? [06:34<00:00, 6.24s/it]
100% 18/18 [05:09<00:00, 4.71s/it]
18/|/| 18/? [05:09<00:00, 4.71s/it]
Epoch 316 Loss => total: 2.0052 , g: 1.9830, l: 0.0068, t: 0.0139, p: 0.0015, theta: 150.378, theta_d: 335.999
100% 18/18 [06:35<00:00, 6.60s/it]
18/|/| 18/? [06:35<00:00, 6.60s/it]
100% 18/18 [05:13<00:00, 5.19s/it]
18/|/| 18/? [05:13<00:00, 5.19s/it]
Epoch 317 Loss => total: 1.9987 , g: 1.9769, l: 0.0068, t: 0.0138, p: 0.0013, theta: 150.395, theta_d: 337.001
100% 18/18 [06:33<00:00, 6.71s/it]
18/|/| 18/? [06:33<00:00, 6.71s/it]
100% 18/18 [05:11<00:00, 4.90s/it]
18/|/| 18/? [05:11<00:00, 4.90s/it]
Epoch 318 Loss => total: 1.9923 , g: 1.9708, l: 0.0067, t: 0.0138, p: 0.0011, theta: 150.411, theta_d: 337.944
100% 18/18 [06:37<00:00, 6.46s/it]
18/|/| 18/? [06:37<00:00, 6.46s/it]
100% 18/18 [05:18<00:00, 5.06s/it]
18/|/| 18/? [05:18<00:00, 5.06s/it]
Epoch 319 Loss => total: 1.9854 , g: 1.9641, l: 0.0067, t: 0.0137, p: 0.0009, theta: 150.425, theta_d: 338.745
100% 18/18 [06:37<00:00, 6.83s/it]
18/|/| 18/? [06:37<00:00, 6.83s/it]
100% 18/18 [05:23<00:00, 5.55s/it]
18/|/| 18/? [05:23<00:00, 5.55s/it]
Epoch 320 Loss => total: 1.9795 , g: 1.9585, l: 0.0067, t: 0.0136, p: 0.0007, theta: 150.439, theta_d: 339.532
100% 18/18 [06:34<00:00, 6.40s/it]
18/|/| 18/? [06:34<00:00, 6.40s/it]
100% 18/18 [05:12<00:00, 4.71s/it]
18/|/| 18/? [05:12<00:00, 4.71s/it]
Epoch 321 Loss => total: 1.9746 , g: 1.9539, l: 0.0067, t: 0.0136, p: 0.0005, theta: 150.452, theta_d: 340.287
100% 18/18 [06:38<00:00, 6.08s/it]
18/|/| 18/? [06:38<00:00, 6.08s/it]
100% 18/18 [05:12<00:00, 5.36s/it]
18/|/| 18/? [05:12<00:00, 5.36s/it]
Epoch 322 Loss => total: 1.9670 , g: 1.9464, l: 0.0066, t: 0.0135, p: 0.0004, theta: 150.465, theta_d: 341.009
100% 18/18 [06:34<00:00, 6.54s/it]
18/|/| 18/? [06:34<00:00, 6.54s/it]
100% 18/18 [05:12<00:00, 6.24s/it]
18/|/| 18/? [05:12<00:00, 6.24s/it]
Epoch 323 Loss => total: 1.9601 , g: 1.9397, l: 0.0066, t: 0.0135, p: 0.0004, theta: 150.477, theta_d: 341.700
100% 18/18 [06:36<00:00, 6.40s/it]
18/|/| 18/? [06:36<00:00, 6.40s/it]
100% 18/18 [05:10<00:00, 5.18s/it]
18/|/| 18/? [05:10<00:00, 5.18s/it]
Epoch 324 Loss => total: 1.9545 , g: 1.9342, l: 0.0066, t: 0.0134, p: 0.0003, theta: 150.489, theta_d: 342.360
100% 18/18 [06:28<00:00, 6.53s/it]
18/|/| 18/? [06:28<00:00, 6.53s/it]
100% 18/18 [05:11<00:00, 4.84s/it]
18/|/| 18/? [05:11<00:00, 4.84s/it]
Epoch 325 Loss => total: 1.9485 , g: 1.9283, l: 0.0065, t: 0.0133, p: 0.0003, theta: 150.500, theta_d: 342.992
100% 18/18 [06:54<00:00, 7.70s/it]
18/|/| 18/? [06:54<00:00, 7.70s/it]
100% 18/18 [05:13<00:00, 4.88s/it]
18/|/| 18/? [05:13<00:00, 4.88s/it]
Epoch 326 Loss => total: 1.9419 , g: 1.9218, l: 0.0065, t: 0.0133, p: 0.0004, theta: 150.510, theta_d: 343.596
100% 18/18 [07:11<00:00, 7.59s/it]
18/|/| 18/? [07:11<00:00, 7.59s/it]
100% 18/18 [05:23<00:00, 5.91s/it]
18/|/| 18/? [05:23<00:00, 5.91s/it]
Epoch 327 Loss => total: 1.9374 , g: 1.9173, l: 0.0065, t: 0.0132, p: 0.0004, theta: 150.520, theta_d: 344.174
100% 18/18 [06:38<00:00, 6.12s/it]
18/|/| 18/? [06:38<00:00, 6.12s/it]
100% 18/18 [05:10<00:00, 5.72s/it]
18/|/| 18/? [05:10<00:00, 5.72s/it]
Epoch 328 Loss => total: 1.9328 , g: 1.9128, l: 0.0064, t: 0.0132, p: 0.0004, theta: 150.530, theta_d: 344.727
100% 18/18 [06:30<00:00, 5.99s/it]
18/|/| 18/? [06:30<00:00, 5.99s/it]
100% 18/18 [05:11<00:00, 5.11s/it]
18/|/| 18/? [05:11<00:00, 5.11s/it]
Epoch 329 Loss => total: 1.9228 , g: 1.9029, l: 0.0064, t: 0.0131, p: 0.0004, theta: 150.539, theta_d: 345.257
100% 18/18 [06:38<00:00, 6.86s/it]
18/|/| 18/? [06:38<00:00, 6.86s/it]
100% 18/18 [05:11<00:00, 5.05s/it]
18/|/| 18/? [05:11<00:00, 5.05s/it]
Epoch 330 Loss => total: 1.9197 , g: 1.8998, l: 0.0064, t: 0.0130, p: 0.0004, theta: 150.548, theta_d: 345.766
100% 18/18 [06:38<00:00, 6.75s/it]
18/|/| 18/? [06:38<00:00, 6.75s/it]
100% 18/18 [05:12<00:00, 5.16s/it]
18/|/| 18/? [05:12<00:00, 5.16s/it]
Epoch 331 Loss => total: 1.9100 , g: 1.8902, l: 0.0064, t: 0.0130, p: 0.0005, theta: 150.557, theta_d: 346.255
100% 18/18 [07:05<00:00, 6.70s/it]
18/|/| 18/? [07:05<00:00, 6.70s/it]
100% 18/18 [05:08<00:00, 4.64s/it]
18/|/| 18/? [05:08<00:00, 4.64s/it]
Epoch 332 Loss => total: 1.9035 , g: 1.8838, l: 0.0063, t: 0.0129, p: 0.0005, theta: 150.565, theta_d: 346.725
100% 18/18 [07:00<00:00, 5.76s/it]
18/|/| 18/? [07:00<00:00, 5.76s/it]
100% 18/18 [05:10<00:00, 5.07s/it]
18/|/| 18/? [05:10<00:00, 5.07s/it]
Epoch 333 Loss => total: 1.9029 , g: 1.8833, l: 0.0063, t: 0.0128, p: 0.0005, theta: 150.573, theta_d: 347.177
100% 18/18 [06:52<00:00, 5.85s/it]
18/|/| 18/? [06:52<00:00, 5.85s/it]
100% 18/18 [05:15<00:00, 5.84s/it]
18/|/| 18/? [05:15<00:00, 5.84s/it]
Epoch 334 Loss => total: 1.8906 , g: 1.8710, l: 0.0063, t: 0.0128, p: 0.0005, theta: 150.580, theta_d: 347.611
100% 18/18 [06:52<00:00, 5.35s/it]
18/|/| 18/? [06:52<00:00, 5.35s/it]
100% 18/18 [05:12<00:00, 5.73s/it]
18/|/| 18/? [05:12<00:00, 5.73s/it]
Epoch 335 Loss => total: 1.8885 , g: 1.8690, l: 0.0062, t: 0.0127, p: 0.0005, theta: 150.588, theta_d: 348.030
100% 18/18 [06:56<00:00, 6.68s/it]
18/|/| 18/? [06:56<00:00, 6.68s/it]
100% 18/18 [05:12<00:00, 5.34s/it]
18/|/| 18/? [05:12<00:00, 5.34s/it]
Epoch 336 Loss => total: 1.8775 , g: 1.8581, l: 0.0062, t: 0.0127, p: 0.0006, theta: 150.595, theta_d: 348.433
100% 18/18 [06:56<00:00, 6.77s/it]
18/|/| 18/? [06:56<00:00, 6.76s/it]
100% 18/18 [05:10<00:00, 4.92s/it]
18/|/| 18/? [05:10<00:00, 4.92s/it]
Epoch 337 Loss => total: 1.8723 , g: 1.8530, l: 0.0062, t: 0.0126, p: 0.0006, theta: 150.601, theta_d: 348.821
100% 18/18 [06:50<00:00, 5.89s/it]
18/|/| 18/? [06:50<00:00, 5.89s/it]
100% 18/18 [05:05<00:00, 4.20s/it]
18/|/| 18/? [05:04<00:00, 4.20s/it]
Epoch 338 Loss => total: 1.8642 , g: 1.8450, l: 0.0061, t: 0.0125, p: 0.0006, theta: 150.608, theta_d: 349.194
100% 18/18 [06:41<00:00, 6.60s/it]
18/|/| 18/? [06:41<00:00, 6.60s/it]
100% 18/18 [05:11<00:00, 5.31s/it]
18/|/| 18/? [05:11<00:00, 5.31s/it]
Epoch 339 Loss => total: 1.8574 , g: 1.8382, l: 0.0061, t: 0.0125, p: 0.0006, theta: 150.614, theta_d: 349.554
100% 18/18 [06:33<00:00, 6.55s/it]
18/|/| 18/? [06:33<00:00, 6.55s/it]
100% 18/18 [05:05<00:00, 5.34s/it]
18/|/| 18/? [05:05<00:00, 5.34s/it]
Epoch 340 Loss => total: 1.8509 , g: 1.8318, l: 0.0061, t: 0.0124, p: 0.0006, theta: 150.620, theta_d: 349.899
100% 18/18 [06:19<00:00, 5.67s/it]
18/|/| 18/? [06:19<00:00, 5.67s/it]
100% 18/18 [05:09<00:00, 5.39s/it]
18/|/| 18/? [05:09<00:00, 5.39s/it]
Epoch 341 Loss => total: 1.8448 , g: 1.8258, l: 0.0061, t: 0.0123, p: 0.0006, theta: 150.626, theta_d: 350.232
100% 18/18 [06:39<00:00, 6.67s/it]
18/|/| 18/? [06:39<00:00, 6.67s/it]
100% 18/18 [05:06<00:00, 5.34s/it]
18/|/| 18/? [05:06<00:00, 5.34s/it]
Epoch 342 Loss => total: 1.8386 , g: 1.8197, l: 0.0060, t: 0.0123, p: 0.0006, theta: 150.632, theta_d: 350.551
100% 18/18 [06:31<00:00, 6.09s/it]
18/|/| 18/? [06:31<00:00, 6.09s/it]
100% 18/18 [05:07<00:00, 5.15s/it]
18/|/| 18/? [05:07<00:00, 5.15s/it]
Epoch 343 Loss => total: 1.8335 , g: 1.8147, l: 0.0060, t: 0.0122, p: 0.0006, theta: 150.637, theta_d: 350.858
100% 18/18 [06:11<00:00, 5.58s/it]
18/|/| 18/? [06:11<00:00, 5.58s/it]
100% 18/18 [05:10<00:00, 4.95s/it]
18/|/| 18/? [05:10<00:00, 4.95s/it]
Epoch 344 Loss => total: 1.8257 , g: 1.8070, l: 0.0060, t: 0.0121, p: 0.0006, theta: 150.642, theta_d: 351.153
100% 18/18 [06:06<00:00, 5.32s/it]
18/|/| 18/? [06:06<00:00, 5.32s/it]
100% 18/18 [05:10<00:00, 17.26s/it]
18/|/| 18/? [05:10<00:00, 17.26s/it]
Epoch 345 Loss => total: 1.8193 , g: 1.8007, l: 0.0059, t: 0.0121, p: 0.0006, theta: 150.647, theta_d: 351.436
100% 18/18 [06:46<00:00, 6.40s/it]
18/|/| 18/? [06:46<00:00, 6.40s/it]
100% 18/18 [05:18<00:00, 5.45s/it]
18/|/| 18/? [05:18<00:00, 5.45s/it]
Epoch 346 Loss => total: 1.8219 , g: 1.8034, l: 0.0059, t: 0.0120, p: 0.0006, theta: 150.652, theta_d: 351.707
100% 18/18 [06:49<00:00, 6.42s/it]
18/|/| 18/? [06:49<00:00, 6.42s/it]
100% 18/18 [05:09<00:00, 4.61s/it]
18/|/| 18/? [05:09<00:00, 4.61s/it]
Epoch 347 Loss => total: 1.8073 , g: 1.7888, l: 0.0059, t: 0.0120, p: 0.0006, theta: 150.656, theta_d: 351.967
100% 18/18 [06:44<00:00, 5.63s/it]
18/|/| 18/? [06:44<00:00, 5.63s/it]
100% 18/18 [05:27<00:00, 5.41s/it]
18/|/| 18/? [05:27<00:00, 5.41s/it]
Epoch 348 Loss => total: 1.8015 , g: 1.7832, l: 0.0058, t: 0.0119, p: 0.0006, theta: 150.661, theta_d: 352.217
100% 18/18 [06:35<00:00, 6.07s/it]
18/|/| 18/? [06:35<00:00, 6.07s/it]
100% 18/18 [05:12<00:00, 5.39s/it]
18/|/| 18/? [05:12<00:00, 5.39s/it]
Epoch 349 Loss => total: 1.7935 , g: 1.7753, l: 0.0058, t: 0.0118, p: 0.0006, theta: 150.665, theta_d: 352.456
100% 18/18 [06:46<00:00, 6.12s/it]
18/|/| 18/? [06:46<00:00, 6.12s/it]
100% 18/18 [05:15<00:00, 4.86s/it]
18/|/| 18/? [05:15<00:00, 4.86s/it]
Epoch 350 Loss => total: 1.7886 , g: 1.7704, l: 0.0058, t: 0.0118, p: 0.0006, theta: 150.669, theta_d: 352.685
100% 18/18 [06:41<00:00, 6.89s/it]
18/|/| 18/? [06:41<00:00, 6.89s/it]
100% 18/18 [05:21<00:00, 6.62s/it]
18/|/| 18/? [05:21<00:00, 6.62s/it]
Epoch 351 Loss => total: 1.7818 , g: 1.7637, l: 0.0058, t: 0.0117, p: 0.0006, theta: 150.673, theta_d: 352.905
100% 18/18 [06:41<00:00, 6.05s/it]
18/|/| 18/? [06:41<00:00, 6.05s/it]
100% 18/18 [05:08<00:00, 5.31s/it]
18/|/| 18/? [05:08<00:00, 5.31s/it]
Epoch 352 Loss => total: 1.7753 , g: 1.7573, l: 0.0057, t: 0.0117, p: 0.0006, theta: 150.676, theta_d: 353.116
100% 18/18 [06:34<00:00, 6.35s/it]
18/|/| 18/? [06:34<00:00, 6.35s/it]
100% 18/18 [05:26<00:00, 6.35s/it]
18/|/| 18/? [05:26<00:00, 6.35s/it]
Epoch 353 Loss => total: 1.7688 , g: 1.7510, l: 0.0057, t: 0.0116, p: 0.0006, theta: 150.680, theta_d: 353.318
100% 18/18 [06:31<00:00, 6.39s/it]
18/|/| 18/? [06:31<00:00, 6.39s/it]
100% 18/18 [05:10<00:00, 5.65s/it]
18/|/| 18/? [05:10<00:00, 5.65s/it]
Epoch 354 Loss => total: 1.7619 , g: 1.7442, l: 0.0057, t: 0.0115, p: 0.0006, theta: 150.683, theta_d: 353.512
100% 18/18 [06:13<00:00, 5.54s/it]
18/|/| 18/? [06:13<00:00, 5.54s/it]
100% 18/18 [05:14<00:00, 5.14s/it]
18/|/| 18/? [05:14<00:00, 5.14s/it]
Epoch 355 Loss => total: 1.7565 , g: 1.7389, l: 0.0056, t: 0.0115, p: 0.0006, theta: 150.686, theta_d: 353.699
100% 18/18 [06:23<00:00, 5.97s/it]
18/|/| 18/? [06:23<00:00, 5.97s/it]
100% 18/18 [05:12<00:00, 4.61s/it]
18/|/| 18/? [05:12<00:00, 4.61s/it]
Epoch 356 Loss => total: 1.7475 , g: 1.7299, l: 0.0056, t: 0.0114, p: 0.0006, theta: 150.690, theta_d: 353.877
100% 18/18 [05:58<00:00, 6.02s/it]
18/|/| 18/? [05:58<00:00, 6.02s/it]
100% 18/18 [04:57<00:00, 4.88s/it]
18/|/| 18/? [04:57<00:00, 4.88s/it]
Epoch 357 Loss => total: 1.7447 , g: 1.7272, l: 0.0056, t: 0.0113, p: 0.0006, theta: 150.693, theta_d: 354.049
100% 18/18 [05:51<00:00, 6.27s/it]
18/|/| 18/? [05:51<00:00, 6.27s/it]
100% 18/18 [04:58<00:00, 4.83s/it]
18/|/| 18/? [04:58<00:00, 4.83s/it]
Epoch 358 Loss => total: 1.7385 , g: 1.7211, l: 0.0055, t: 0.0113, p: 0.0006, theta: 150.695, theta_d: 354.214
100% 18/18 [05:48<00:00, 6.57s/it]
18/|/| 18/? [05:48<00:00, 6.57s/it]
100% 18/18 [04:55<00:00, 4.68s/it]
18/|/| 18/? [04:55<00:00, 4.68s/it]
Epoch 359 Loss => total: 1.7328 , g: 1.7155, l: 0.0055, t: 0.0112, p: 0.0006, theta: 150.698, theta_d: 354.372
100% 18/18 [05:53<00:00, 6.33s/it]
18/|/| 18/? [05:53<00:00, 6.33s/it]
100% 18/18 [04:59<00:00, 5.36s/it]
18/|/| 18/? [04:59<00:00, 5.36s/it]
Epoch 360 Loss => total: 1.7268 , g: 1.7096, l: 0.0055, t: 0.0112, p: 0.0006, theta: 150.701, theta_d: 354.525
100% 18/18 [05:53<00:00, 5.96s/it]
18/|/| 18/? [05:53<00:00, 5.96s/it]
100% 18/18 [05:04<00:00, 4.97s/it]
18/|/| 18/? [05:04<00:00, 4.97s/it]
Epoch 361 Loss => total: 1.7209 , g: 1.7038, l: 0.0055, t: 0.0111, p: 0.0006, theta: 150.703, theta_d: 354.671
100% 18/18 [06:00<00:00, 5.89s/it]
18/|/| 18/? [06:00<00:00, 5.89s/it]
100% 18/18 [05:13<00:00, 6.74s/it]
18/|/| 18/? [05:13<00:00, 6.74s/it]
Epoch 362 Loss => total: 1.7261 , g: 1.7091, l: 0.0054, t: 0.0111, p: 0.0006, theta: 150.706, theta_d: 354.812
100% 18/18 [06:16<00:00, 7.92s/it]
18/|/| 18/? [06:16<00:00, 7.92s/it]
100% 18/18 [05:12<00:00, 7.18s/it]
18/|/| 18/? [05:12<00:00, 7.18s/it]
Epoch 363 Loss => total: 1.7185 , g: 1.7016, l: 0.0054, t: 0.0110, p: 0.0006, theta: 150.708, theta_d: 354.947
100% 18/18 [06:11<00:00, 7.77s/it]
18/|/| 18/? [06:11<00:00, 7.77s/it]
100% 18/18 [05:07<00:00, 5.20s/it]
18/|/| 18/? [05:07<00:00, 5.20s/it]
Epoch 364 Loss => total: 1.7133 , g: 1.6964, l: 0.0054, t: 0.0109, p: 0.0006, theta: 150.711, theta_d: 355.077
100% 18/18 [06:07<00:00, 5.58s/it]
18/|/| 18/? [06:07<00:00, 5.58s/it]
100% 18/18 [05:19<00:00, 6.61s/it]
18/|/| 18/? [05:19<00:00, 6.61s/it]
Epoch 365 Loss => total: 1.7044 , g: 1.6876, l: 0.0054, t: 0.0109, p: 0.0006, theta: 150.713, theta_d: 355.203
100% 18/18 [06:10<00:00, 6.81s/it]
18/|/| 18/? [06:10<00:00, 6.81s/it]
100% 18/18 [05:05<00:00, 16.98s/it]
18/|/| 18/? [05:05<00:00, 16.98s/it]
Epoch 366 Loss => total: 1.7016 , g: 1.6849, l: 0.0053, t: 0.0108, p: 0.0006, theta: 150.715, theta_d: 355.324
100% 18/18 [06:07<00:00, 5.64s/it]
18/|/| 18/? [06:07<00:00, 5.64s/it]
100% 18/18 [05:07<00:00, 4.59s/it]
18/|/| 18/? [05:07<00:00, 4.59s/it]
Epoch 367 Loss => total: 1.6901 , g: 1.6734, l: 0.0053, t: 0.0108, p: 0.0006, theta: 150.717, theta_d: 355.440
100% 18/18 [06:09<00:00, 7.35s/it]
18/|/| 18/? [06:09<00:00, 7.34s/it]
100% 18/18 [05:09<00:00, 4.85s/it]
18/|/| 18/? [05:09<00:00, 4.85s/it]
Epoch 368 Loss => total: 1.6883 , g: 1.6717, l: 0.0053, t: 0.0107, p: 0.0006, theta: 150.719, theta_d: 355.552
100% 18/18 [06:09<00:00, 6.21s/it]
18/|/| 18/? [06:09<00:00, 6.21s/it]
100% 18/18 [05:07<00:00, 4.78s/it]
18/|/| 18/? [05:07<00:00, 4.78s/it]
Epoch 369 Loss => total: 1.6851 , g: 1.6686, l: 0.0052, t: 0.0107, p: 0.0006, theta: 150.721, theta_d: 355.660
100% 18/18 [06:08<00:00, 5.96s/it]
18/|/| 18/? [06:08<00:00, 5.96s/it]
100% 18/18 [05:30<00:00, 6.97s/it]
18/|/| 18/? [05:30<00:00, 6.97s/it]
Epoch 370 Loss => total: 1.6727 , g: 1.6563, l: 0.0052, t: 0.0106, p: 0.0006, theta: 150.723, theta_d: 355.764
100% 18/18 [06:33<00:00, 6.16s/it]
18/|/| 18/? [06:33<00:00, 6.16s/it]
100% 18/18 [05:26<00:00, 6.38s/it]
18/|/| 18/? [05:26<00:00, 6.38s/it]
Epoch 371 Loss => total: 1.6720 , g: 1.6557, l: 0.0052, t: 0.0105, p: 0.0006, theta: 150.724, theta_d: 355.865
100% 18/18 [06:27<00:00, 6.40s/it]
18/|/| 18/? [06:27<00:00, 6.40s/it]
100% 18/18 [05:28<00:00, 5.60s/it]
18/|/| 18/? [05:28<00:00, 5.60s/it]
Epoch 372 Loss => total: 1.6574 , g: 1.6412, l: 0.0052, t: 0.0105, p: 0.0006, theta: 150.726, theta_d: 355.962
100% 18/18 [06:24<00:00, 7.43s/it]
18/|/| 18/? [06:24<00:00, 7.43s/it]
100% 18/18 [05:16<00:00, 5.50s/it]
18/|/| 18/? [05:16<00:00, 5.50s/it]
Epoch 373 Loss => total: 1.6593 , g: 1.6432, l: 0.0051, t: 0.0104, p: 0.0006, theta: 150.728, theta_d: 356.056
100% 18/18 [06:22<00:00, 8.46s/it]
18/|/| 18/? [06:22<00:00, 8.46s/it]
100% 18/18 [05:21<00:00, 5.80s/it]
18/|/| 18/? [05:21<00:00, 5.80s/it]
Epoch 374 Loss => total: 1.6554 , g: 1.6393, l: 0.0051, t: 0.0104, p: 0.0006, theta: 150.729, theta_d: 356.146
100% 18/18 [06:11<00:00, 5.94s/it]
18/|/| 18/? [06:11<00:00, 5.94s/it]
100% 18/18 [05:22<00:00, 6.28s/it]
18/|/| 18/? [05:21<00:00, 6.28s/it]
Epoch 375 Loss => total: 1.6449 , g: 1.6289, l: 0.0051, t: 0.0103, p: 0.0006, theta: 150.731, theta_d: 356.233
100% 18/18 [06:35<00:00, 6.66s/it]
18/|/| 18/? [06:35<00:00, 6.66s/it]
100% 18/18 [05:34<00:00, 6.68s/it]
18/|/| 18/? [05:34<00:00, 6.68s/it]
Epoch 376 Loss => total: 1.6365 , g: 1.6206, l: 0.0051, t: 0.0103, p: 0.0006, theta: 150.732, theta_d: 356.318
100% 18/18 [06:14<00:00, 5.61s/it]
18/|/| 18/? [06:14<00:00, 5.61s/it]
100% 18/18 [05:17<00:00, 5.62s/it]
18/|/| 18/? [05:17<00:00, 5.62s/it]
Epoch 377 Loss => total: 1.6339 , g: 1.6180, l: 0.0050, t: 0.0102, p: 0.0006, theta: 150.734, theta_d: 356.399
100% 18/18 [06:21<00:00, 8.48s/it]
18/|/| 18/? [06:21<00:00, 8.48s/it]
100% 18/18 [05:26<00:00, 5.14s/it]
18/|/| 18/? [05:26<00:00, 5.13s/it]
Epoch 378 Loss => total: 1.6281 , g: 1.6123, l: 0.0050, t: 0.0102, p: 0.0006, theta: 150.735, theta_d: 356.478
100% 18/18 [06:22<00:00, 5.88s/it]
18/|/| 18/? [06:22<00:00, 5.88s/it]
100% 18/18 [05:29<00:00, 6.69s/it]
18/|/| 18/? [05:29<00:00, 6.69s/it]
Epoch 379 Loss => total: 1.6217 , g: 1.6060, l: 0.0050, t: 0.0101, p: 0.0006, theta: 150.736, theta_d: 356.555
100% 18/18 [06:07<00:00, 6.45s/it]
18/|/| 18/? [06:07<00:00, 6.45s/it]
100% 18/18 [05:07<00:00, 4.77s/it]
18/|/| 18/? [05:07<00:00, 4.77s/it]
Epoch 380 Loss => total: 1.6162 , g: 1.6006, l: 0.0050, t: 0.0101, p: 0.0006, theta: 150.738, theta_d: 356.629
100% 18/18 [05:59<00:00, 5.69s/it]
18/|/| 18/? [05:59<00:00, 5.69s/it]
100% 18/18 [05:04<00:00, 5.25s/it]
18/|/| 18/? [05:04<00:00, 5.25s/it]
Epoch 381 Loss => total: 1.6131 , g: 1.5976, l: 0.0049, t: 0.0100, p: 0.0006, theta: 150.739, theta_d: 356.700
100% 18/18 [06:01<00:00, 6.76s/it]
18/|/| 18/? [06:01<00:00, 6.76s/it]
100% 18/18 [05:05<00:00, 4.87s/it]
18/|/| 18/? [05:05<00:00, 4.87s/it]
Epoch 382 Loss => total: 1.6051 , g: 1.5896, l: 0.0049, t: 0.0100, p: 0.0006, theta: 150.740, theta_d: 356.770
100% 18/18 [05:58<00:00, 6.77s/it]
18/|/| 18/? [05:58<00:00, 6.77s/it]
100% 18/18 [05:11<00:00, 6.29s/it]
18/|/| 18/? [05:11<00:00, 6.29s/it]
Epoch 383 Loss => total: 1.6002 , g: 1.5848, l: 0.0049, t: 0.0099, p: 0.0006, theta: 150.741, theta_d: 356.837
100% 18/18 [06:28<00:00, 6.47s/it]
18/|/| 18/? [06:28<00:00, 6.47s/it]
100% 18/18 [05:12<00:00, 5.54s/it]
18/|/| 18/? [05:12<00:00, 5.54s/it]
Epoch 384 Loss => total: 1.5945 , g: 1.5791, l: 0.0049, t: 0.0099, p: 0.0006, theta: 150.742, theta_d: 356.902
100% 18/18 [05:54<00:00, 5.63s/it]
18/|/| 18/? [05:54<00:00, 5.63s/it]
100% 18/18 [05:02<00:00, 5.53s/it]
18/|/| 18/? [05:02<00:00, 5.53s/it]
Epoch 385 Loss => total: 1.5925 , g: 1.5772, l: 0.0048, t: 0.0098, p: 0.0006, theta: 150.743, theta_d: 356.965
100% 18/18 [05:54<00:00, 7.17s/it]
18/|/| 18/? [05:54<00:00, 7.17s/it]
100% 18/18 [05:00<00:00, 16.67s/it]
18/|/| 18/? [05:00<00:00, 16.67s/it]
Epoch 386 Loss => total: 1.5847 , g: 1.5695, l: 0.0048, t: 0.0098, p: 0.0006, theta: 150.745, theta_d: 357.026
100% 18/18 [06:06<00:00, 8.25s/it]
18/|/| 18/? [06:06<00:00, 8.25s/it]
100% 18/18 [05:10<00:00, 5.00s/it]
18/|/| 18/? [05:10<00:00, 5.00s/it]
Epoch 387 Loss => total: 1.5799 , g: 1.5648, l: 0.0048, t: 0.0097, p: 0.0006, theta: 150.746, theta_d: 357.085
100% 18/18 [06:08<00:00, 6.19s/it]
18/|/| 18/? [06:08<00:00, 6.19s/it]
100% 18/18 [05:11<00:00, 5.16s/it]
18/|/| 18/? [05:11<00:00, 5.16s/it]
Epoch 388 Loss => total: 1.5769 , g: 1.5618, l: 0.0048, t: 0.0097, p: 0.0006, theta: 150.747, theta_d: 357.143
'''



