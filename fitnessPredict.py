from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from generateArtich import Generator,randomSpaceSelector
class fitnessPredict():
    def __init__(self):

         #turn the space into one-hot
        self.rf=RandomForestRegressor(n_estimators=1000)

    def predict(self,x_test_net,x_train_net,y_train_net):
        #encode
        x_train=list()
        for i in range(len(x_train_net)):
               x_train.append(self.turn_dict_to_numpy(self.process_x(x_train_net[i])))
        x_test=list()
        for i in range(len(x_test_net)):
               x_test.append(self.turn_dict_to_numpy(self.process_x(x_test_net[i])))
        return self.train_rf(np.array(x_train),np.array(y_train_net),np.array(x_test))


    def process_x(self,net):
        net_nb_layer=net.get('nb_layers')
        Search_space = randomSpaceSelector()
        max_np_layers=Search_space.get_max_nb_dense_layers()
        # Mutate one of the params.
        one_hot_dict=dict()
        for key in net.keys():
            if key=='nb_layers':
                pass
            else:
                net_p=net.get(key)
                if isinstance(net_p, list):
                    if len(net_p)>net_nb_layer:
                        net_p=net_p[:net_nb_layer]
                    if isinstance(net_p[0],str):

                        space=Search_space.get_sapce_for_bayes(key)
                        _=list()
                        for i in range(len(space)):
                            _.append(0)
                        one_hot_list=list()
                        for i in range(len(net_p)):
                            __=list(_)
                            __[space.index(net_p[i])]=1
                            one_hot_list.append(__)
                        for i in range(max_np_layers-len(net_p)):
                            one_hot_list.append(list(_))
                        one_hot_dict[key]=one_hot_list
                    else:

                        for i in range(max_np_layers - len(net_p)):
                            net_p.append(0)

                        one_hot_dict[key] = net_p
                else:
                    if isinstance(net_p, str):
                        space = Search_space.get_sapce_for_bayes(key)
                        _ = list()
                        for i in range(len(space)):
                            _.append(0)
                        _[space.index(net_p)] = 1
                        one_hot_dict[key]=_
                    else:
                        one_hot_dict[key] = net_p
        return one_hot_dict

    def turn_dict_to_numpy(self,net):
        _=list()
        for key in net.keys():
            net_p = net.get(key)
            if isinstance(net_p, list):
                for i in range (len(net_p)):
                    if isinstance(net_p[i], list):
                        for j in range(len(net_p[i])):
                            _.append(net_p[i][j])
                    else:
                        _.append(net_p[i])
            else:
                _.append(net_p)
        return _

    def train_rf(self,x,y,x_val):
        self.rf.fit(x, y)
        print('rf score:{}'.format(self.rf.score(x, y)))
        return list(self.rf.predict(x_val))

