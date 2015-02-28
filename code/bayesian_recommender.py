import numpy as np
import math

class Feature(object):
    """The features of an item."""

    def __init__(self, name,n,m):
        self.name = name
        # n: the num of this feature has been used to decribe an item
        # m: the total num of items
        # prior probability
        self.p = (n+0.5)/(m+1)
        # inverted document frequency(idf)
        self.idf = math.log(m*1.0/n+1)
        self.init_prob_ev()

    def init_prob_ev(self):
        self.pev = self.p
        # self.relevance = False

    def set_prob_ev(self, w):
        self.pev = self.p + w*self.p*(1-self.p)
        # self.relevance = True


class Item(object):
    """An item."""

    def __init__(self, name, parents):
        self.name = name
        self.parents = parents
        self.pa_size = parents.size
        self.weights = np.zeros(self.pa_size)
        self.init_weights()

    def init_weights(self):
        for i in range(self.pa_size):
            self.weights[i] = self.parents[i].idf
        self.weights = self.weights/np.sum(self.weights)
        # self.relevance = False

    # set p(feature|item as evidence), propagate to features
    def prob_f_iev(self):
        for i in range(self.pa_size):
            self.parents[i].set_prob_ev(self.weights[i])

    # propagate from features to item
    def set_prob_ev(self):
        for i in range(self.pa_size):
            self.pev += self.parents[i].pev*self.weights[i]
            # if self.parents[i].relevance:
            #     self.relevance = True


class User(object):
    """A user."""

    def __init__(self, name, pa_id_ra):
        self.name = name
        # parents_id_rating table
        self.pa_id_ra = pa_id_ra
        # neighbor_similarity
        # self.ne_sim = ne_sim
        self.pa_size = pa_id_ra.size
        # self.ne_size = ne_sim.size
        # content-based probability given evidence
        self.cb_pev = np.zeros(6)
        self.cf_pev = np.zeros(6)

    # propagate from items to user
    def set_cb_prob_ev(self, predict_item):
        if predict_item not in self.pa_id_ra['id']:
            for i in range(self.pa_size):
                pa = self.pa_id_ra[i]['parent']
                self.cb_pev[0] += (1-pa.pev)
                r = self.pa_id_ra[i]['rating']
                self.cb_pev[r] += pa.pev
            self.cb_pev = self.pev/np.sum(self.pev)
        else:
            r = self.pa_id_ra['rating'][np.where(self.pa_id_ra['id']==predict_item)]
            self.cb_pev[r] = 1

    # compute the probability of A rating with a value s when Ui rated with t
    # P(A=s|Ui=t)
    def set_neighbor_pattern(self):
        rating_pattern = np.zeros((self.ne_size,5,5), dtype=int)
        for i in range(self.ne_size):
            ne = self.ne_sim['neighbor'][i]
            same_parents_self = self.pa_id_ra['rating'][np.in1d(self.pa_id_ra['id'], ne.pa_id_ra['id'])]
            same_parents_neig = ne.pa_id_ra['rating'][np.in1d(ne.pa_id_ra['id'], self.pa_id_ra['id'])]
            for j in range(same_parents_self.size):
                rating_pattern[i,same_parents_neig[j],same_parents_self[j]] += 1
        rating_pattern = (rating_pattern+1.0/5)/(np.sum(rating_pattern,axis=-1,keepdims=True)+1)
        self.weights = self.rating_pattern*self.ne_sim['sim'][:,np.newaxis,np.newaxis]

    def set_cf_prob_ev(self):
        weights = np.copy(self.weights)
        w_r0 = np.copy(self.ne_sim['sim'])
        # propagate from neighbors to Acf
        for i in range(self.ne_size):
            weights[i] *= self.ne_sim['neighbor'][i].cb_pev[1:][:,np.newaxis]
            w_r0[i] *= self.ne_sim['neighbor'][i].cb_pev[0]
        # sum all the neighbors for ratings 1~5
        self.cf_pev[1:] = np.sum(np.sum(weights,axis=1),axis=0)
        self.cf_pev[0] = np.sum(w_r0)
        self.cf_pev = self.cf_pev/np.sum(self.cf_pev)

    def set_hybrid_prob_ev(self):
        pass



class BayesianRecommender(object):
    """A bayesian network based hybrid recommender system."""

    def __init__(self, item_file, rating_file, info_file):
        self.rating_file = rating_file
        self.item_file = item_file
        self.info_file = info_file
        self.init_nodes()



    def init_nodes(self):
        self.read_data()
        self.create_features()
        self.create_items()
        self.create_users()

    def read_data(self):
        self.item_features = np.loadtxt(self.item_file,delimiter='|',dtype='int8', usecols=range(5,24))
        self.user_item_rating = np.loadtxt(self.rating_file,dtype='int16',usecols=range(3))
        self.info = np.loadtxt(self.info_file, dtype=('int,S10'))
        self.total_features = self.item_features.shape[1]
        # self.total_items = self.user_item_rating.size
        self.total_items = self.info['f0'][1]
        self.total_users = self.info['f0'][0]

    def create_features(self):
        # # count the times of rating of each item
        # items_id, item_rated_counts = np.unique(self.user_item_rating[:,1],return_counts=True)
        # print items_id, item_rated_counts
        # feature_counts = np.copy(self.item_features)[items_id-1]*item_rated_counts[:,np.newaxis]
        # feature_counts = np.sum(feature_counts,axis=0)
 
        # compute the number of each feature that has been used to describe an item
        feature_counts = np.sum(self.item_features,axis=0)
        # print feature_counts
        # create nodes for each feature
        self.features = np.empty(self.total_features,dtype='O')
        for i in range(self.total_features):
            self.features[i] = Feature(i,feature_counts[i],self.total_items)

    def create_items(self):
        self.items = np.empty(self.total_items,dtype='O')
        for i in range(self.total_items):
            parents = self.features[self.item_features[i]==1]
            self.items[i] = Item(i,parents)

    def create_users(self):
        self.users = np.empty(self.total_users,dtype='O')
        _, users_first_index, users_rating_counts = np.unique(self.user_item_rating[:,0], return_index=True, return_counts=True)
        for i in range(self.total_users):
            start = users_first_index[i]
            end = start + users_rating_counts[i]
            pa_ids = self.user_item_rating[start:end,1]
            pa_id_ra = np.empty(users_rating_counts[i],dtype=[('parent','O'),('id','int'),('rating','int8')])
            pa_id_ra['parent'] = self.items[pa_ids-1]
            pa_id_ra['id'] = self.user_item_rating[start:end,1]
            pa_id_ra['rating'] = self.user_item_rating[start:end,2]
            self.users[i] = User(i,pa_id_ra)

    def inference(self, user, item):
        pass
        # set all features probs
        # set all items probs
        # set all users cb probs
        # set neighbors' cf prob
        # set final cb-cf prob




def read_item_features(item_file):
    item_f = np.loadtxt(item_file,delimiter='|',dtype='int8', usecols=range(5,24))
    return item_f

def read_rating(rating_file):
    user_item_rating = np.loadtxt(rating_file,dtype='int16',usecols=range(3))
    return user_item_rating

def read_info(info_file):
    info = np.loadtxt(info_file, dtype=('int,a10'))
    return info

def feature_test(files):
    br = BayesianRecommender(*files)
    return br


if __name__ == '__main__':
    # f = read_item_features('../data/ml-100k/u.item')
    # r = read_rating('../data/ml-100k/u1.base')
    # info = read_info('../data/ml-100k/u.info')
    files = ['../data/ml-100k/u.item','../data/ml-100k/u1.base','../data/ml-100k/u.info']
    result = feature_test(files)






















