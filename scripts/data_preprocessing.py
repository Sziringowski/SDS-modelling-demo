#import database_management as db_management
from database_management import database
import documentation as doc
import copy
import numpy as np
import uuid
import re
import pandas as pd
import itertools
import glob
import pymc as pm
import arviz as az
import xarray as xr
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter


def compare (key: tuple, parametr: str, fix_params: dict) -> bool:
    i = 0
    while str(parametr) != str(list(fix_params.keys())[i]):
        if str(key[i]) != str(list(fix_params.values())[i]):
            return 0
            break
        i += 1
    i += 1
    while i < len(list(fix_params.keys())):
        if str(key[i]) != str(list(fix_params.values())[i]):
            return 0
            break
        i += 1
    return 1

def key_enum (param: str, doc=doc.features_space):
    i = 0
    while i < len(list(doc.keys())):
        if str(param) == str(list(doc.keys())[i]):
            return i
            break
        i += 1
    return 'ERROR'


class file_data_proccesor:
    def __init__(self, file_path: str, database: database):
        self.path = file_path  # путь до фалйа
        self.Data = dict()  # таргет атрибут класса!
        self.uploaded_strings = []  # добавленные записи
        self.correct_number_of_sizes = 0  # кол-во замеров
        self.database = database  # инстанс дб
        self.cleared_data = []  # очищенная дата
        self.removed_data = []  # удалённые точки

    def _read(self):
        body_template = doc.body_template
        with open(self.path, 'r', encoding='utf-8') as file:
            ###
            lines = file.readlines()
            data = []
            for line in lines:
                data.append( line.strip().split(' ') )


            file_protection = [temp_data[0].split('=')[1].upper() for temp_data in data if 'protection' in temp_data[0]][0]
    
            file_HW_chassis = [temp_data[0].split('=')[1].upper() for temp_data in data if 'HW_chassis' in temp_data[0]][0]

            file_SSD = [int(temp_data[0].split('=')[1]) for temp_data in data if 'SSD' in temp_data[0]][0]
    
            i = 0
            counter = 0
            self.correct_number_of_sizes += str(data).count('###')
            # realization of data reading functions
            while i < len(data):
                
                if '###' in data[i]:
                    index = 0
                    counter += 1

                    body_size = copy.deepcopy(body_template)
                    
                    Params = dict.fromkeys(list(doc.features_space.keys()))
                    for temp in list( Params.keys() ):
                        Params[temp] = 0
                    
                    temp_data_0 = [k for k in data[i] if len(k) > 0]

                    id_size = str(uuid.uuid4()) 

                    #  записываем конфиг замера
                    body_size["system"]["path"] = self.path
                    body_size["parametrs"]["protection"] = file_protection
                    body_size["parametrs"]["nodes"] = int(temp_data_0[1])
                    body_size["parametrs"]["ssd"] = file_SSD
                    body_size["parametrs"]["op_type"] =  temp_data_0[4].lower()
                    body_size["parametrs"]["hw_chassis"] = file_HW_chassis
                    
                    if re.fullmatch( r'(\b\d*kb\b)', str( temp_data_0[3] ).lower()):  # size like kb
                        body_size["parametrs"]["size"] = int(temp_data_0[3][:-2])
                    elif re.fullmatch( r'(\b\d*kib\b)', str( temp_data_0[3] ).lower() ):  #size like kib
                        body_size["parametrs"]["size"] = int(temp_data_0[3][:-3])
                    elif re.fullmatch( r'(\b\d*mb\b)', str( temp_data_0[3] ).lower() ):  #size like mb
                        body_size["parametrs"]["size"] = int(temp_data_0[3][:-2]) * 1000  # convert to KB
                    elif re.fullmatch( r'(\b\d*mib\b)', str( temp_data_0[3] ).lower() ):  #size like mib
                        body_size["parametrs"]["size"] = int(temp_data_0[3][:-3]) * 1000
                    else:
                        body_size["parametrs"]["size"] = 'INCORRECT DATA'

                    i += 1
                    
                    # пишем фичи замера
                    while i < len(data) - 1 and not('###' in data[i+1]):
                        
                        if 'aws_obj_get_bytes' in str(data[i]) or 'aws_obj_put_bytes' in str(data[i]):
                            temp_data = [k for k in data[i] if len(k) > 0]
                            if str(temp_data[-1]).lower() == 'mb/s':
                                body_size["values"]["straight"]["mbps"].append(float(temp_data[-2]))
                            elif str(temp_data[-1]).lower() == 'gb/s':
                                body_size["values"]["straight"]["mbps"].append(float(temp_data[-2]) * 1000)  # convert to mb/s
                            else:
                                body_size["values"]["straight"]["mbps"].append('INCORRECT DATA')
                        
                        if 'aws_obj_get_duration' in str(data[i]) or 'aws_obj_put_duration' in str(data[i]):
                            temp_data = [k.split('=') for k in data[i] if len(k) > 0]
                            for j in range(1, 6+1):
                                if temp_data[j][0] == "p(90)":
                                    temp_data[j][0] = "p90"
                                elif temp_data[j][0] == "p(95)":
                                    temp_data[j][0] = "p95"


                            for j in range(1,6+1):                            
                                if 'ms' in str(temp_data[j][1]).lower():  # case M.SECONDS
                                    body_size["values"]["straight"][temp_data[j][0]].append(float(temp_data[j][1][:-2]))  # we suppose that notation <digits>m<digits><MS> is impossible 
                                        
                                else:  # case SECONDS 
                                    match = re.fullmatch( r'(\b\d*\dm\d\d*\w*\b)', str(temp_data[j][1]).lower() )  # <digits (at least 1)>m<digits (at least 1)><s> - found m in d[b][1] in seconds
                                    if match:
                                        body_size["values"]["straight"][temp_data[j][0]].append((float(str(temp_data[j][1][:-1]).split('m')[0]) * 60 + float(str(temp_data[j][1][:-1]).split('m')[1])) * 1000)  # convert to ms| found m
                                    else:
                                        body_size["values"]["straight"][temp_data[j][0]].append(float(temp_data[j][1][:-1]) * 1000)  # convert to ms
                
                        if 'aws_obj_put_success' in str(data[i]) or 'aws_obj_get_success' in str(data[i]):
                            temp_data = [k for k in data[i] if len(k) > 0]
                            body_size["values"]["straight"]['opps'].append(float(temp_data[2][:-2:]))  #obj/s
                            
                
                        if 'aws_obj_get_fails' in str(data[i]):
                            temp_data = [k for k in data[i] if len(k) > 0]
                            body_size["values"]["straight"]['opps_loss'].append(float(temp_data[2][:-2:]))  #obj/s
                        i += 1
                    ### found '###' or end of file, update our Data                    
                    
                    # realization of agregate functions 
                    body_size["values"]["agregated"]['sum_mbps'] = np.sum(body_size["values"]["straight"]['mbps'])
                    body_size["values"]["agregated"]['sum_opps'] = np.sum(body_size["values"]["straight"]['opps'])
                    
                    ###
                    self.Data[id_size] = body_size
                    body_size = {}
                i += 1

    @staticmethod
    def _founder(Data):
        '''
        Функция реально зависит только от Даты: именно состояние замеров в репе является ключевым объектом работы скрипта
        '''
        Broken_sizes = []

        for parametr in doc.cont_features:

            doc_wout_param = copy.deepcopy(doc.features_space)

            doc_wout_param.pop(parametr)

            cross_doc = list(itertools.product(*list(doc_wout_param.values())))

            n = key_enum(parametr)
            cross_doc = [cross_doc[i][:n] + tuple([0]) + cross_doc[i][n:] for i in range(len(cross_doc))]

            cross_doc_dict = [
                {
                    list(doc.features_space.keys())[i]: item[i] for i in range(len(list(doc.features_space.keys())))
                } for item in cross_doc

            ]  # костыльно добавляем нуль в валью парама, чтобы не изменять функцию compare

            for i in range(len(cross_doc_dict)):

                X = np.array([
                    [tuple(Data[id]['parametrs'].values())[key_enum(parametr)], id] for id in list(Data.keys()) if compare(tuple(Data[id]['parametrs'].values()), parametr, cross_doc_dict[i])
                    ], dtype=object)
                
                Y = np.array([
                    Data[id]['values']['agregated']['sum_opps'] for id in list(Data.keys()) if compare(tuple(Data[id]['parametrs'].values()), parametr, cross_doc_dict[i])
                    ], dtype=object)

                W = np.column_stack((X, Y))   # (cont., id, target value)
                
                if parametr == "nodes":
                    W = W[W[:, 0].argsort()].T
                elif parametr == "size":
                    W = W[W[:, 0].argsort()[::-1]].T  # ревёрсим массив для сайза, по БЛ при увеличении сайза, уменьшаются таргеты
                for j in range(len(W[0]) - 1):
                    if W[2][j] > W[2][j+1]:
                        if parametr == "nodes":
                            Broken_sizes.append(
                                W[1][j+1]
                            )
                        else:
                            Broken_sizes.append(
                                W[1][j+1]
                            )
                        # в broken_sizes хранятся айдишки

        return {"len": len(Broken_sizes), "broken_sizes": Broken_sizes}

    def _auto_cleaner(self, Data) -> dict:  # запускаем эту функцию, она сама обновит атрибуты
        #Data_operated = copy.deepcopy(Data)
        removed_data = {}
        while self._founder(Data)["len"] > 0:
            for id in self._founder(Data)["broken_sizes"]:
                temp = Data.pop(id, None)
                if temp is not None:
                    removed_data[id] = temp['values']['agregated']["sum_opps"]
        self.cleared_data = Data  # очищенное
        self.removed_data = removed_data  # удалённое

    def write(self):
        self._read()  # читаем
        Data = copy.deepcopy(self.Data)
        self._auto_cleaner(Data)  # чистим
        ids = []
        body_template = doc.body_template
        df_parced = pd.DataFrame(body_template["system"] | body_template["parametrs"] | body_template["values"]["straight"] | body_template["values"]["agregated"])
        for id in list(self.cleared_data.keys()):
            values_temp = {key: value for key, value in self.cleared_data[id]["values"]["straight"].items()}
            new_row = self.cleared_data[id]["system"] | self.cleared_data[id]["parametrs"] | values_temp | self.cleared_data[id]["values"]["agregated"]
            if 'INCORRECT DATA' not in list(new_row.values()): 
                self.database.PUT(values={"id": str(id)} | new_row, table_name="sizes")
                ids.append(id)
            #else ... можно добавить функциональность подсветки неправильных значений
        for id in ids:
            len_t = self.database.GET(table_name="sizes", targets=["id"], features={"id": str(id)})
            if len(len_t) != 0:
                self.uploaded_strings.append(id)
       
class data_instance:  # renamed from `preprocessing`
    def __init__(self, data_df, database: database):
        self.data_df = data_df
        self.data_df_ohe = pd.DataFrame()
        self.database = database
        self.encoded_features = []

        # выполним функции при создании инстанса
        self.ohe()
    

    def correlation_matrix(self):
        correlation_matrix = self.data_df_ohe[self.encoded_features+doc.cont_features].corr()
        plt.figure(figsize=(8, 8))
        hmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        hmap.set_xticklabels(hmap.get_xmajorticklabels(), rotation=45)
        plt.title('Корреляционная матрица')
        plt.show()
        sns.reset_orig()

    def correlation_matrix_to_target(self):
        correlation_matrix = pd.DataFrame(self.data_df_ohe[self.encoded_features+doc.cont_features+[doc.target]].corr()[doc.target]).sort_values(by=[doc.target],ascending=False)
        plt.figure(figsize=(6, 6))
        hmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=False)
        hmap.set_xticklabels(hmap.get_xmajorticklabels(), rotation=0)
        plt.title('Корреляционная матрица')
        plt.show()
        sns.reset_orig()

    def ohe(self):
        data = np.array(self.database.GET(targets=["id"]+doc.features+[doc.target], features={}, table_name="sizes"))
        data_df = pd.DataFrame(data[:,1:], index=data[:,0], columns=doc.features+[doc.target])
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        data_df_ohe = ohe.fit_transform(data_df[doc.cat_features])
        encoded_columns = ohe.get_feature_names_out(doc.cat_features)
        data_df_ohe = pd.DataFrame(data_df_ohe, columns=encoded_columns)
        data_df_ohe = pd.concat([data_df.drop(doc.cat_features,axis=1,inplace=False).reset_index(drop=True), data_df_ohe.reset_index(drop=True)], axis=1)
        data_df_ohe.index = data_df.index
        data_df_ohe["sum_opps"] = data_df_ohe["sum_opps"].astype('float')
        data_df_ohe[[col for col in data_df_ohe.columns if col != doc.target]] = data_df_ohe[[col for col in data_df_ohe if col != doc.target]].astype('int')
        self.data_df_ohe = data_df_ohe
        self.encoded_features = list(encoded_columns)

    def distribution(self):  
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20), sharex=False)
        axes = axes.flatten()

        for i, feature in enumerate([col for col in self.data_df_ohe.columns if col != doc.target]):
            switcher_disc = False if feature in doc.cont_features else True
            sns.histplot(self.data_df_ohe[feature], kde=(not switcher_disc), ax=axes[i], color='skyblue', edgecolor='black', stat='density', discrete=switcher_disc)
            axes[i].set_xlabel("значение")
            axes[i].set_title(f'распределение {feature}')
            axes[i].grid(True, linestyle="--", alpha=0.7)

        plt.ylabel("фича")
        plt.tight_layout()
        plt.show()

    def compare_graphs(self, parametr: str, fix_params_1: dict, fix_params_2: dict):
        ''' example of fix_params_i:
        fix_params = {
        'protection': '*',
        'size': '*',
        'nodes': '*',
        'op_type': '*',
        'hw_chassis': '*',
        'ssd': '*'
        }
        '''
        data_1 = np.array(self.database.GET(table_name="sizes", targets=[parametr]+[doc.target],features=fix_params_1),dtype=float)
        data_2 = np.array(self.database.GET(table_name="sizes", targets=[parametr]+[doc.target],features=fix_params_2),dtype=float)
        W_1 = data_1
        W_2 = data_2
        W_1 = W_1[W_1[:, 0].argsort()].T; W_2 = W_2[W_2[:, 0].argsort()].T
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharex=False)
        axes = axes.flatten()

        for i, W in enumerate([W_1, W_2]):
            axes[i].scatter(W[0], W[1], color='r', alpha = 1, s=10)
            #
            W_t = np.array([[0,0]])
            for item in W.T:
                if item[0] not in W_t[:,0]:
                    W_t = np.append(W_t,[item],axis=0)
            W_t = W_t[1:].T
            #
            axes[i].plot(W_t[0], W_t[1],'k-', alpha = 1)
            axes[i].set_xlabel(parametr)
            axes[i].set_title(f'opps для набора №{i+1}')
            axes[i].grid(True, linestyle="--", alpha=0.7)
    

        graph_title = {key: {'first graph': fix_params_1[key], 'second graph': fix_params_2[key]} for key in [x for x in list(doc.features_space.keys()) if x != parametr]}
        #graph_title.pop(parametr, None)
        print(f'params:\n{pd.DataFrame(graph_title).transpose()}')
        plt.ylabel("opps")
        plt.tight_layout()
        plt.show()
        


