import numpy as np

class Feature(object):
    """The features of an item."""

    def __init__(self, name,n,m):
        self.name = name
        # n: the num of this feature has been used to decribe an item
        # m: the total num of items
        # prior probability
        self.p = (n+0.5)/(m+1)
        # inverted document frequency(idf)
        self.idf = np.log(m*1.0/n+1)
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
    def set_prob_f_iev(self):
        for i in range(self.pa_size):
            self.parents[i].set_prob_ev(self.weights[i])

    # propagate from features to item
    def set_prob_ev(self):
        self.pev = 0
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
        self.pa_size = pa_id_ra.size
        # content-based probability given evidence
        self.cb_pev = np.zeros(6)
        self.cf_pev = np.zeros(6)
        self.hb_pev = None

    def add_neighbors(self, ne_sim):
        # neighbor_similarity
        self.ne_sim = ne_sim
        self.ne_size = ne_sim.size
        self.set_neighbor_pattern()

    # propagate from items to user
    def set_cb_prob_ev(self, predict_item):
        if predict_item not in self.pa_id_ra['id']:
            for i in range(self.pa_size):
                pa = self.pa_id_ra[i]['parent']
                self.cb_pev[0] += (1-pa.pev)
                r = self.pa_id_ra[i]['rating']
                self.cb_pev[r] += pa.pev
            self.cb_pev = self.cb_pev/np.sum(self.cb_pev)
        else:
            r = self.pa_id_ra['rating'][np.where(self.pa_id_ra['id']==predict_item)]
            for i in range(6):
                if i != r:
                    # print i
                    self.cb_pev[i] = 0
                else:
                    self.cb_pev[i] = 1

    # compute the probability of A rating with a value s when Ui rated with t
    # P(A=s|Ui=t)
    def set_neighbor_pattern(self):
        rating_pattern = np.zeros((self.ne_size,5,5), dtype=int)
        for i in range(self.ne_size):
            ne = self.ne_sim['neighbor'][i]
            same_parents_self = self.pa_id_ra['rating'][np.in1d(self.pa_id_ra['id'], ne.pa_id_ra['id'])]
            same_parents_neig = ne.pa_id_ra['rating'][np.in1d(ne.pa_id_ra['id'], self.pa_id_ra['id'])]
            for j in range(same_parents_self.size):
                rating_pattern[i,same_parents_neig[j]-1,same_parents_self[j]-1] += 1
        rating_pattern = (rating_pattern+1.0/5)/(np.sum(rating_pattern,axis=-1,keepdims=True)+1)
        self.weights = rating_pattern*self.ne_sim['sim'][:,np.newaxis,np.newaxis]

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

    def set_hybrid_prob_ev(self,alpha):
        # very very compact computation for hybrid probability
        # alpha = self.cf_pev[0]
        matrix = self.cb_pev*self.cf_pev.reshape((6,1))
        # print matrix
        # print 'cb',self.cb_pev
        # print 'cf',self.cf_pev
        cb_sum = np.sum(matrix,axis=-1) - matrix.diagonal()
        # print cb_sum
        cf_sum = np.sum(matrix,axis=0) - matrix.diagonal()
        # print cf_sum
        self.hb_pev = matrix.diagonal() + cb_sum*alpha + cf_sum*(1-alpha)
        self.hb_pev = self.hb_pev/np.sum(self.hb_pev)
        # print self.hb_pev

class Demographic(object):
    """Demographic info: age, gender, occupation."""

    def __init__(self, name,n,m):
        self.name = name
        # n: the num of this demographic has been used to decribe an user
        # m: the total num of user
        # prior probability
        self.p = (n+0.5)/(m+1)
        # inverted document frequency(idf)
        self.idf = np.log(m*1.0/n+1)
        self.init_prob_ev()

    def init_prob_ev(self):
        self.pev = self.p
        # self.relevance = False

class BayesianRecommender(object):
    """A bayesian network based hybrid recommender system."""

    def __init__(self, item_file, rating_file, info_file, test_file, top_k_ne):
        self.rating_file = rating_file
        self.item_file = item_file
        self.info_file = info_file
        self.test_file = test_file
        self.top_k_ne = top_k_ne
        self.init_nodes()

    def init_nodes(self):
        self.read_data()
        self.create_features()
        self.create_items()
        self.create_users()
        self.create_neighbors()

    def read_data(self):
        self.item_features = np.loadtxt(self.item_file,delimiter='|',dtype='int8', usecols=range(5,24))
        self.user_item_rating = np.loadtxt(self.rating_file,dtype='int16',usecols=range(3))
        self.info = np.loadtxt(self.info_file, dtype=('int,S10'))
        self.total_features = self.item_features.shape[1]
        self.total_items = self.info['f0'][1]
        self.total_users = self.info['f0'][0]
        self.test_data = np.loadtxt(self.test_file,dtype='int16',usecols=range(3))
        self.test_size = self.test_data.shape[0]

    def create_features(self): 
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
            pa_id_ra['id'] = self.user_item_rating[start:end,1]-1
            pa_id_ra['rating'] = self.user_item_rating[start:end,2]
            self.users[i] = User(i,pa_id_ra)

    def create_neighbors(self):
        sim_matirx = np.zeros((self.total_users,self.total_users))-2
        for i in range(self.total_users):
            # if i > 1:
            #     break
            # # print i
            for j in range(i+1,self.total_users):
                ui_pa = self.users[i].pa_id_ra
                uj_pa = self.users[j].pa_id_ra
                ui_ratings = ui_pa['rating'][np.in1d(ui_pa['id'],uj_pa['id'])]
                if ui_ratings.size:
                    uj_ratings = uj_pa['rating'][np.in1d(uj_pa['id'],ui_pa['id'])]
                    # uj = uj_pa['rating'][np.in1d(uj_pa['id'],ui_pa['id'])] - np.mean(uj_pa['rating'])
                    ui_aver_ra = np.mean(ui_pa['rating'])
                    uj_aver_ra = np.mean(uj_pa['rating'])
                    # Peason correlation coefficient
                    ui = ui_ratings - ui_aver_ra
                    uj = uj_ratings - uj_aver_ra
                    sim_ij = np.sum(ui*uj)/np.sqrt(np.sum(np.square(ui))*np.sum(np.square(uj)))*ui.size/ui_pa.size
                else:
                    sim_ij = 0
                if np.isnan(sim_ij):
                    sim_ij = 0
                sim_matirx[i][j] = abs(sim_ij)
        sim_matirx = sim_matirx + np.triu(sim_matirx,1).T
        top_ne_index = np.argsort(sim_matirx,axis=1)[:,-self.top_k_ne:]

        for i in range(self.total_users):
            ne_sim = np.empty(self.top_k_ne,dtype=[('neighbor','O'),('sim','f')])
            ne_sim['neighbor'] = self.users[np.sort(top_ne_index[i])]
            ne_sim['sim'] = sim_matirx[i][top_ne_index[i]]
            self.users[i].add_neighbors(ne_sim)


    def inference(self, user_id, item_id, alp):
        user = self.users[user_id]
        item = self.items[item_id]
        # set all features probs
        for f in self.features:
            f.init_prob_ev()
        item.set_prob_f_iev()
        # set all items probs
        for i in self.items:
            i.set_prob_ev()
        # set all users cb probs
        for u in self.users:
            u.set_cb_prob_ev(item_id)
        # set neighbors' cf prob
        user.set_cf_prob_ev()
        # set final cb-cf prob
        user.set_hybrid_prob_ev(alp)

        # return user

    def testing(self,alp):
        self.result = np.zeros(self.test_size)
        self.result_cb = np.zeros(self.test_size)
        for i in range(self.test_size):
        # for i in range(1):
            if i%1000 == 0:
                print i
            userid = self.test_data[i,0]-1
            itemid = self.test_data[i,1]-1
            self.inference(userid, itemid, alp)
            self.result[i] = np.argmax(self.users[userid].hb_pev[1:]) + 1
            self.result_cb[i] = np.argmax(self.users[userid].cb_pev[1:]) + 1
        self.mae = np.sum(np.fabs(self.result - self.test_data[:,2]))*1.0/self.test_size
        self.error = np.count_nonzero(self.result - self.test_data[:,2])*1.0/self.test_size
        self.cb_mae = np.sum(np.fabs(self.result_cb - self.test_data[:,2]))*1.0/self.test_size
        self.cb_error = np.count_nonzero(self.result_cb - self.test_data[:,2])*1.0/self.test_size

        # print self.result[:10]
        # print self.test_data[:10,2]
        # print self.mae
        # print self.error

def read_item_features(item_file):
    item_f = np.loadtxt(item_file,delimiter='|',dtype='int8', usecols=range(5,24))
    return item_f

def read_rating(rating_file):
    user_item_rating = np.loadtxt(rating_file,dtype='int16',usecols=range(3))
    return user_item_rating

def read_info(info_file):
    info = np.loadtxt(info_file, dtype=('int,a10'))
    return info

def test(files):
    br = BayesianRecommender(*files)
    return br

class CrossValidation(object):
    """5-fold cross validation."""

    def __init__(self):
        self.folds = 5
        self.top_k_ne = [10,20,30,50]
        self.mae = np.zeros((5,4))
        self.error = np.zeros((5,4))
        self.cb_mae = np.zeros((5,4))
        self.cb_error = np.zeros((5,4))
        self.path = '../data/ml-100k/u'

    def run(self):
        for i in range(self.folds):
            k = str(i+1)
            print '\nfold:',k
            for idx,j in enumerate(self.top_k_ne):
                print '\ntop_k_ne:',j
                files =  [self.path+'.item', self.path+k+'.base', self.path+'.info', self.path+k+'.test',j]
                br = BayesianRecommender(*files)
                br.testing()
                print 'mae:',br.mae
                print 'error:',br.error
                print 'cb_mae:',br.cb_mae
                print 'cb_error:',br.cb_error
                self.mae[i,idx] = br.mae
                self.error[i,idx] = br.error
                self.cb_mae[i,idx] = br.cb_mae
                self.cb_error[i,idx] = br.cb_error
        self.aver_mae = np.mean(self.mae,axis=0)
        self.aver_error = np.mean(self.error,axis=0)
        self.cb_aver_mae = np.mean(self.cb_mae,axis=0)
        self.cb_aver_error = np.mean(self.cb_error,axis=0)
        print
        print '\nmae:\n',self.mae, 
        print '\nerror\n',self.error
        print '\ncb_mae:\n',self.cb_mae, 
        print '\ncb_error\n',self.cb_error
        print self.aver_mae, self.aver_error
        print self.cb_aver_mae, self.cb_aver_error


# def cross_validation():
#     folds = 5
#     top_k_ne = [10,20,30,50]
#     mae = np.zeros((5,4))
#     error = np.zeros((5,4))
#     for i in range(folds):
#         for j in top_k_ne:
#             files =  ['../data/ml-100k/u.item','../data/ml-100k/u'+str(i)+'.base','../data/ml-100k/u.info','../data/ml-100k/u'+str(i)+'.test',j]
#             br = BayesianRecommender(*files)
#             br.testing()
#             mae[i,j] = br.mae
#             error[i,j] = br.error
#     self.aver_mae = np.mean(mae,axis=0)
#     aver_error = np.mean(error,axis=0)
#     print mae
#     print error
#     print aver_mae, aver_error
#     return 

if __name__ == '__main__':

    files = ['../data/ml-100k/u.item','../data/ml-100k/u1.base','../data/ml-100k/u.info','../data/ml-100k/u1.test',10]
    re = test(files) 
    print 'finish create model, start inference'
    re.testing(0.3)
    # u = re.inference(1,6)

    # test = CrossValidation()
    # result = test.run()


