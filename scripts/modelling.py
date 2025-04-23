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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from collections import Counter
from sklearn.linear_model import RidgeCV, LinearRegression, Ridge
from typing import List
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool
import optuna
import json_tricks

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping





class grid:
    def __init__(self, database: database, iterations = 1, circle_accept=10):
        self.database = database  # инстанс бд
        self.lines = []
        self.iterations = iterations  # кол-во полных циклов генерации синтетики iterations=0 значит не генерим ничего
        self.circle_accept = circle_accept # кол-во допустимых циклов при генерации
        #

        for iter_num in range(self.iterations):
            self.lines.extend(self._synth_generation(iter_num+1))

    def _synth_generation(self, iter_num):
        print(f'iteration {iter_num} / {self.iterations}')
        all_values = np.array(self.database.GET(targets=["protection", "size", "op_type", "hw_chassis", "ssd"], features={}, table_name="sizes", distinct=True))
        L = len(all_values)
        circle_lines = []

        # генерим синт вдоль нодов
        for index_row, row in enumerate(all_values):
            fix_params = dict(zip(["protection", "size", "op_type", "hw_chassis", "ssd"], row))
            # получаем
            Z = np.array(self.database.GET(targets=["nodes", "sum_opps"], features=fix_params, table_name="sizes"))
            Z[:,0] = Z[:,0].astype(int); Z[:,1] = Z[:,1].astype(float)
            # разбили
            Z = Z[Z[:, 0].argsort()].T
            X = Z[0] # nodes из БД
            X = np.array([[np.log(X[i])**3, np.log(X[i])**2, np.log(X[i])] for i in range(len(X))])  # after coordinates transf
            Y = np.array(Z[1])  # target value из БД

            switсher_while = True
            circle_count = 0  # подсчитываем, сколько раз прошли через while
            while switсher_while:
                temporary_lines = []
                circle_count += 1
                if circle_count >= self.circle_accept: 
                    print(f'''troubles with:\nfeatures: {fix_params}\n{pd.DataFrame(Z.T, columns=["nodes", "target"])}'''); break

                # модель
                ridge_alpha = np.random.uniform(0.001, 1)
                ridge_cv = Ridge(alpha=ridge_alpha)  # alpha = 0.1
                ridge_cv.fit(X, Y)

                predictions = np.array([ridge_cv.predict([x])[0] for x in X])  # на том, что было в БД

                # метрики на трейне (не было разбиения, проверяем адекватность)
                mse = mean_squared_error(Y, predictions)
                mae = mean_absolute_error(Y, predictions)
                r2 = ridge_cv.score(X, predictions)

                #range_for_gen = np.linspace(4, 100, 30, dtype=int)
                range_for_gen = np.random.randint(4, 100, size=40, dtype=int)

                # генерим синтетику
                for n in range_for_gen:
                    id_line = "line-" + str(uuid.uuid4())
                    
                    fix_params_t = copy.deepcopy(fix_params)

                    value = ridge_cv.predict([[np.log(n)**3, np.log(n)**2, np.log(n)]])
                    
                    values_t = {
                        "id": id_line,
                        "protection": fix_params_t["protection"],
                        "size": int(fix_params_t["size"]),
                        "nodes": n,
                        "op_type": fix_params_t["op_type"],
                        "hw_chassis": fix_params_t["hw_chassis"],
                        "ssd": int(fix_params_t["ssd"]),
                        "sum_opps": value[0],
                        "r2": r2,
                        "mse": mse,
                        "mae": mae
                    }
                    temporary_lines.append(values_t)
                # к этому моменту всё нагенерено для этого набора парамов вдоль нод

                W = np.array([[item["nodes"], item["sum_opps"]] for item in temporary_lines], dtype=float)
                condition = np.array_equal(W[W[:,0].argsort()], W[W[:,1].argsort()])# and r2 > 0.6  # проверка на то, что наши opps возрастают вдоль нод и метрики вменяемые

                if condition:
                    circle_lines.extend(temporary_lines)
                    print(f'generated {index_row+1} / {L} of lines with ridge_alpha={ridge_alpha:.2f}, r2={r2:.2f}, mae={mae:.2f} ({iter_num})')
                    switсher_while = False; break
        
        print('\nnodes generated\n')
        return circle_lines

    def write(self, method="database", validation=True):  # БД наворачивается в ошибку памяти
        if method=="database":
            for ind, item in enumerate(self.lines):
                self.database.PUT(values=item, table_name="sizes_synthetic", validation=validation)
                print(f'{ind/len(self.lines)*100:.2f}%')
        elif method=="json":
            random_key = str(uuid.uuid4())
            with open(f'synthetic\synth_lines_{random_key}.json', 'w', encoding='utf-8') as f:
                json_tricks.dump(self.lines, f, ensure_ascii=False, indent=4)
                print(f'file saved as: \nsynth_lines_{random_key}.json')
        else: 
            print("incorrect method")

    def read(self, json_file):
        with open(f'synthetic\{json_file}', 'r', encoding='utf-8') as f:
            data_loaded = json_tricks.load(f)
            self.lines.extend(data_loaded)

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
        #вариант из БД
        #data_1 = np.array(self.database.GET(table_name="sizes_synthetic", targets=[parametr]+[doc.target],features=fix_params_1),dtype=float)
        #data_2 = np.array(self.database.GET(table_name="sizes_synthetic", targets=[parametr]+[doc.target],features=fix_params_2),dtype=float)

        # вариант из self.lines
        data_1 = np.array([[item[parametr], item[doc.target]] for item in self.lines if all(item[key] == fix_params_1[key] for key in fix_params_1.keys())])
        data_2 = np.array([[item[parametr], item[doc.target]] for item in self.lines if all(item[key] == fix_params_2[key] for key in fix_params_2.keys())])

        W_1 = data_1
        W_2 = data_2
        W_1 = W_1[W_1[:, 0].argsort()].T; W_2 = W_2[W_2[:, 0].argsort()].T
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharex=False)
        axes = axes.flatten()

        for i, W in enumerate([W_1, W_2]):
            axes[i].scatter(W[0], W[1], color='grey', alpha = 1, s=10)
            #
  
            W_t = [[i, []] for i in (np.unique(W[0]))]
            for j, h in enumerate(np.unique(W[0])):
                W_t[j][1].extend([w[1] for w in W.T if w[0] == h])
            W_t = np.array([[w[0], np.mean(w[1])] for w in W_t]).T
            W_t[0] = W_t[0].astype(int)
            #
            axes[i].plot(W_t[0].astype(int), W_t[1],'k-', alpha = 1)
            axes[i].set_xlabel(parametr)
            axes[i].set_title(f'opps для набора синтетики №{i+1}')
            axes[i].grid(True, linestyle="--", alpha=0.7)
    

        graph_title = {key: {'first graph': fix_params_1[key], 'second graph': fix_params_2[key]} for key in [x for x in list(doc.features_space.keys()) if x != parametr]}
        #graph_title.pop(parametr, None)
        print(f'params:\n{pd.DataFrame(graph_title).transpose()}')
        plt.ylabel("opps")
        plt.tight_layout()
        plt.show()

    def GET(self, targets: List[str], features: dict, lines='self', distinct=True):
        try:
            lines_to_search = []
            if lines == 'self':
                lines_to_search = self.lines
            elif type(lines)==list and len(lines)>0 and type(lines[0])==dict:
                lines_to_search = lines
            if len(targets) != 0:
                output = [
                    [item[t] for t in targets] for item in lines_to_search if all(item[key] == features[key] for key in features.keys())
                    ]
            elif len(targets)==0 or targets==[]:
                output = [
                    list(item.values()) for item in lines_to_search if all(item[key] == features[key] for key in features.keys())
                    ]
            return output if (not distinct) else pd.DataFrame(output).drop_duplicates().to_numpy()
        
        except Exception as e:
            print(f'Exception {e}')
    

class catboost_model:
    def __init__(self, n_trials, data_df_prototyping, data_df_learn, cv = 5, seed=42):
        self.n_trials = n_trials  # колво итераций прототипирования
        self.data_df_prototyping = data_df_prototyping  # датасет для прототипировани optune
        self.seed = seed
        self.data_df_learn = data_df_learn  # датасет для обучения модели после протипирования
        self.best_params = {"parametrs": 0, "value": 0}
        self.model_metrics = {}
        self.cv = cv
        self.model = lambda x: x  # обученная модель

        print(f"created CatBoostRegressor instance\nlength of prototyping data = {len(self.data_df_prototyping)}\nlength of learn data = {len(self.data_df_learn)}\nseed = {self.seed}")


    def prototyping(self, path_to_upload="optimal_params_1.xlsx"):
        current_df = self.data_df_prototyping.copy()
        #======= SCALING
        current_df["size"] = np.log(np.array(current_df["size"], dtype=float))
        #======= FEATURES ENGINEERING
        current_df["size/nodes"] = np.array(current_df["size"], dtype=float) / np.array(current_df["nodes"], dtype=float)
        current_df["ssd/size"] = np.array(current_df["ssd"], dtype=float) / np.array(current_df["size"], dtype=float)
        added_f = ["size/nodes", "ssd/size"]
        #=======
        X = current_df[doc.features+added_f]
        Y = current_df[doc.target].fillna(0)

        X[doc.cat_features] = X[doc.cat_features].fillna(0).astype("str")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)

        def objective(trial):

            params = {
                'iterations': trial.suggest_int('iterations', 100, 1700),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 2, 10),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 100, 600),
                'model_size_reg': trial.suggest_float('model_size_reg', 0, 1),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0, 1),
                'loss_function': 'MAE',
                'rsm': trial.suggest_float("rsm", 0, 1),
                'verbose': False,
                'random_seed': self.seed,
                "eval_metric": "MAE"
            }

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            rmse_scores = []
            r2_scores = []
            mae_scores = []


            for train_index, valid_index in kf.split(X_train):
                X_kf_train, X_kf_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
                y_kf_train, y_kf_valid = Y_train.iloc[train_index], Y_train.iloc[valid_index]

                model = CatBoostRegressor(**params, cat_features=doc.cat_features, thread_count=1)
                model.fit(X_kf_train, y_kf_train, eval_set=(X_kf_valid, y_kf_valid), early_stopping_rounds=100, verbose=1)
                
                preds = model.predict(X_kf_valid)
                rmse = np.sqrt(mean_squared_error(y_kf_valid, preds))
                rmse_scores.append(rmse)
                r2 = r2_score(y_kf_valid, preds)
                r2_scores.append(r2)
                mae = mean_absolute_error(y_kf_valid, preds)
                mae_scores.append(mae)
            return np.mean(mae_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)

        best_params = {}
        best_params["parametrs"] = study.best_params
        best_params["value"] = study.best_value

        pd.DataFrame(best_params).to_excel(path_to_upload, header=True)

        self.best_params = best_params

    def train(self, test_rate=0.2):
        current_df = self.data_df_learn.copy()
        #======= SCALING
        current_df["size"] = np.log(np.array(current_df["size"], dtype=float))
        current_df["size/nodes"] = np.array(current_df["size"], dtype=float) / np.array(current_df["nodes"], dtype=float)
        current_df["ssd/size"] = np.array(current_df["ssd"], dtype=float) / np.array(current_df["size"], dtype=float)
        added_f = ["size/nodes", "ssd/size"]

        X = current_df[doc.features+added_f]
        Y = current_df[doc.target].fillna(0)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_rate, random_state=self.seed)

        model = CatBoostRegressor(**self.best_params["parametrs"], cat_features=doc.cat_features)
        model.fit(X_train, Y_train, early_stopping_rounds=100, verbose=1)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        r2 = r2_score(Y_test, preds)
        mae = mean_absolute_error(Y_test, preds)
        self.model_metrics = {
            "r2": r2, "mae": mae, "rmse": rmse
        }
        self.model = model

    def predict(self, X: pd.DataFrame):
        current_df = X.copy()
        #======= SCALING
        current_df["size"] = np.log(np.array(current_df["size"], dtype=float))
        current_df["size/nodes"] = np.array(current_df["size"], dtype=float) / np.array(current_df["nodes"], dtype=float)
        current_df["ssd/size"] = np.array(current_df["ssd"], dtype=float) / np.array(current_df["size"], dtype=float)
        added_f = ["size/nodes", "ssd/size"]
        X_pred = current_df[doc.features+added_f]
        predicts = self.model.predict(X_pred)
        W = X.copy()
        W[doc.target] = predicts
        return W

    def compare_graphs(self, parametr: str, fix_params: dict, grid: grid, database: database):
        ''' example of fix_params:
        fix_params = {
        'protection': '*',
        'size': '*',
        'nodes': '*',
        'op_type': '*',
        'hw_chassis': '*',
        'ssd': '*'
        }
        '''
        '''
        хотим уметь сравнивать предсказание с синтетикой (Г1) и с сырой датой и синтетикой (Г2): синтетику выделить слабо в (Г2)
        '''
        #вариант из БД
        #data_1 = np.array(self.database.GET(table_name="sizes_synthetic", targets=[parametr]+[doc.target],features=fix_params_1),dtype=float)
        #data_2 = np.array(self.database.GET(table_name="sizes_synthetic", targets=[parametr]+[doc.target],features=fix_params_2),dtype=float)

        # дата из синтетики
        data_synth = np.array([[item[parametr], item[doc.target]] for item in grid.lines if all(item[key] == fix_params[key] for key in fix_params.keys())])
        # дата из БД
        data_raw = np.array(database.GET(table_name="sizes", targets=[parametr]+[doc.target],features=fix_params), dtype=float)
        # предсказанные значения
        arr = [list(fix_params.values()) + [n] for n in range(4,101)]

        pdfr = pd.DataFrame(arr, columns=list(fix_params.keys())+["nodes"])
        data_predicted = self.predict(pdfr)[[parametr, doc.target]]
        W_synth = data_synth
        W_raw = data_raw
        W_predicted = np.array(data_predicted)

        W_synth = W_synth[W_synth[:, 0].argsort()].T if len(W_synth)!=0 else []
        W_raw = W_raw[W_raw[:, 0].argsort()].T if len(W_raw)!=0 else []
        W_predicted = W_predicted[W_predicted[:, 0].argsort()].T
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharex=False)
        axes = axes.flatten()

        # ax = 0
        if len(W_synth)!=0:
            axes[0].scatter(W_synth[0], W_synth[1], color='yellow', alpha = 0.7, s=10, label="synth")  # вся синтетика
        
            W_t = [[i, []] for i in (np.unique(W_synth[0]))]
            for j, h in enumerate(np.unique(W_synth[0])):
                W_t[j][1].extend([w[1] for w in W_synth.T if w[0] == h])
            W_t = np.array([[w[0], np.mean(w[1])] for w in W_t]).T
            W_t[0] = W_t[0].astype(int)
            #
            axes[0].plot(W_t[0].astype(int), W_t[1],'y-', alpha = 0.8)  # средняя по синетике
        axes[0].plot(W_predicted[0].astype(int), W_predicted[1],'k-', alpha = 1, label="pred")  # предсказанные
        axes[0].set_xlabel(parametr)
        axes[0].set_title(f'opps для предсказания/синтетики; масштаб: 100')
        axes[0].grid(True, linestyle="--", alpha=0.7)
        axes[0].legend()

        # ax = 1
        if len(W_synth)!=0:
            W_synth = W_synth.T[:13].T
            axes[1].scatter(W_synth[0], W_synth[1], color='yellow', alpha = 0.7, s=10, label="synth")  # вся синтетика
            W_t = [[i, []] for i in (np.unique(W_synth[0]))]
            for j, h in enumerate(np.unique(W_synth[0])):
                W_t[j][1].extend([w[1] for w in W_synth.T if w[0] == h])
            W_t = np.array([[w[0], np.mean(w[1])] for w in W_t]).T
            W_t[0] = W_t[0].astype(int)
            #
            axes[1].plot(W_t[0].astype(int), W_t[1],'y-', alpha = 0.8)  # средняя по синетике
        if len(W_raw)!=0:
            axes[1].scatter(W_raw[0], W_raw[1], color='red', alpha = 0.7, s=10, label="raw")  # вся сырая дата
        
        axes[1].plot(W_predicted[0].astype(int), W_predicted[1],'k-', alpha = 1, label="pred")  # предсказанные
        axes[1].set_xlabel(parametr)
        axes[1].set_xlim(3,17)
        axes[1].set_title(f'opps для предсказания/синтетики/сырая; масштаб: 16')
        axes[1].grid(True, linestyle="--", alpha=0.7)
        axes[1].legend()

        graph_title = {key: {'first graph': fix_params[key]} for key in [x for x in list(doc.features_space.keys()) if x != parametr]}
        #graph_title.pop(parametr, None)
        print(f'params:\n{pd.DataFrame(graph_title).transpose()}')
        plt.ylabel("opps")
        plt.tight_layout()
        plt.show()
        

class NN_model:
    def __init__(self, n_trials, data_df_prototyping, data_df_learn, seed=42):
        self.n_trials = n_trials  # колво итераций прототипирования
        self.data_df_prototyping = data_df_prototyping  # датасет для прототипировани optune
        self.seed = seed
        self.data_df_learn = data_df_learn  # датасет для обучения модели после протипирования
        self.best_params = {"parametrs": 0, "value": 0}
        self.model_metrics = {}
        self.model = lambda x: x  # обученная модель
        self.encoder = lambda x: x

        print(f"created Neural Network instance\nlength of prototyping data = {len(self.data_df_prototyping)}\nlength of learn data = {len(self.data_df_learn)}\nseed = {self.seed}")

    def prototyping(self, fit_params, path_to_upload="optimal_params_NN.xlsx"):
        current_df = self.data_df_prototyping.copy()
        #======= SCALING
        current_df["size"] = np.log(np.array(current_df["size"], dtype=float))

        #======= FEATURES ENGINEERING
        current_df["size/nodes"] = np.array(current_df["size"], dtype=float) / np.array(current_df["nodes"], dtype=float)
        current_df["ssd/size"] = np.array(current_df["ssd"], dtype=float) / np.array(current_df["size"], dtype=float)
        added_f = ["size/nodes", "ssd/size"]
        #=======
        X = current_df[doc.features+added_f]
        self.encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        encoded = self.encoder.fit_transform(X[doc.cat_features])
        encoded = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out())
        X = pd.concat([X[doc.cont_features+added_f].reset_index(drop=True),encoded.reset_index(drop=True)], axis=1)
        Y = current_df[doc.target].fillna(0)

        X = np.array(X, dtype=float); Y = np.array(Y, dtype=float)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)



                
        # конфигурируем модель
        def build_and_compile_model(norm):
            model = Sequential([
                norm,
                Dense(32, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1)
                ])

            model.compile(loss='mean_absolute_error',
                            optimizer=tf.keras.optimizers.Adam(0.001))
            return model

        normalizer = tf.keras.layers.Normalization(axis=-1)

        def plot_loss(history):
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            #plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Error [MPG]')
            plt.grid(True)
            plt.legend()
            plt.show()

        normalizer.adapt(np.array(X_train))

        self.model = build_and_compile_model(normalizer)

        early_stopping = EarlyStopping(
            monitor='val_loss',  # следить за метрикой валидации
            patience=10,      # число эпох без улучшения, после которых обучение остановится
            restore_best_weights=True  # восстановить лучшие веса
            )

        history = self.model.fit(
            X_train,
            Y_train,
            callbacks=[early_stopping],
            **fit_params
            )


        rmse_scores = []
        r2_scores = []
        mae_scores = []

        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        rmse_scores.append(rmse)
        r2 = r2_score(Y_test, preds)  # Вычисляем R^2
        r2_scores.append(r2)
        mae = mean_absolute_error(Y_test, preds)
        mae_scores.append(mae)
        self.model_metrics = {
            "r2": r2, "mae": mae, "rmse": rmse
        }

        self.model.save('model_NN\model.h5')
        plot_loss(history)
        print(self.model.summary())


    def train(self, test_rate=0.2):
        current_df = self.data_df_learn.copy()
        #======= FEATURES ENGINEERING
        current_df["size/nodes"] = np.array(current_df["size"], dtype=float) / np.array(current_df["nodes"], dtype=float)
        current_df["ssd/size"] = np.array(current_df["ssd"], dtype=float) / np.array(current_df["size"], dtype=float)
        added_f = ["size/nodes", "ssd/size"]
        #=======
        X = current_df[doc.features+added_f]
        encoder = OneHotEncoder(sparse_output=True, drop="first")
        encoded = encoder.fit_transform(X[doc.cat_features])
        encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
        X = pd.concat([X[doc.cont_features+added_f].reset_index(drop=True),encoded.reset_index(drop=True)], axis=1)
        Y = current_df[doc.target].fillna(0)

        X = np.array(X, dtype=float); Y = np.array(Y, dtype=float)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_rate, random_state=self.seed)

        model = build_and_compile_model(normalizer)
        model.fit(X_train, Y_train, early_stopping_rounds=100, verbose=1)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        r2 = r2_score(Y_test, preds)
        mae = mean_absolute_error(Y_test, preds)
        self.model_metrics = {
            "r2": r2, "mae": mae, "rmse": rmse
        }
        self.model = model

    def predict(self, X: pd.DataFrame):
        current_df = X.copy()
        #======= SCALING
        current_df["size"] = np.log(np.array(current_df["size"], dtype=float))
        #======= FE
        current_df["size/nodes"] = np.array(current_df["size"], dtype=float) / np.array(current_df["nodes"], dtype=float)
        current_df["ssd/size"] = np.array(current_df["ssd"], dtype=float) / np.array(current_df["size"], dtype=float)
        added_f = ["size/nodes", "ssd/size"]
        X_pred = current_df[doc.features+added_f]
        encoded = self.encoder.transform(X_pred[doc.cat_features].astype(str))
        encoded = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out())
        X_pred = pd.concat([X_pred[doc.cont_features+added_f].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)

        X_pred = np.array(X_pred, dtype=float)
        predicts = self.model.predict(X_pred)
        W = X.copy()
        W[doc.target] = predicts
        return W

    def compare_graphs(self, parametr: str, fix_params: dict, grid: grid, database: database):
        ''' example of fix_params:
        fix_params = {
        'protection': '*',
        'size': '*',
        'nodes': '*',
        'op_type': '*',
        'hw_chassis': '*',
        'ssd': '*'
        }
        '''
        '''
        хотим уметь сравнивать предсказание с синтетикой (Г1) и с сырой датой и синтетикой (Г2): синтетику выделить слабо в (Г2)
        '''
        #вариант из БД
        #data_1 = np.array(self.database.GET(table_name="sizes_synthetic", targets=[parametr]+[doc.target],features=fix_params_1),dtype=float)
        #data_2 = np.array(self.database.GET(table_name="sizes_synthetic", targets=[parametr]+[doc.target],features=fix_params_2),dtype=float)

        # дата из синтетики
        data_synth = np.array([[item[parametr], item[doc.target]] for item in grid.lines if all(item[key] == fix_params[key] for key in fix_params.keys())])
        # дата из БД
        data_raw = np.array(database.GET(table_name="sizes", targets=[parametr]+[doc.target],features=fix_params), dtype=float)
        # предсказанные значения
        arr = [list(fix_params.values()) + [n] for n in range(4,101)]
 
        pdfr = pd.DataFrame(arr, columns=list(fix_params.keys())+["nodes"])
        data_predicted = self.predict(pdfr)[[parametr, doc.target]]
        W_synth = data_synth
        W_raw = data_raw
        W_predicted = np.array(data_predicted)

        W_synth = W_synth[W_synth[:, 0].argsort()].T if len(W_synth)!=0 else []
        W_raw = W_raw[W_raw[:, 0].argsort()].T if len(W_raw)!=0 else []
        W_predicted = W_predicted[W_predicted[:, 0].argsort()].T
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharex=False)
        axes = axes.flatten()

        # ax = 0
        if len(W_synth)!=0:
            axes[0].scatter(W_synth[0], W_synth[1], color='yellow', alpha = 0.7, s=10, label="synth")  # вся синтетика
        
            W_t = [[i, []] for i in (np.unique(W_synth[0]))]
            for j, h in enumerate(np.unique(W_synth[0])):
                W_t[j][1].extend([w[1] for w in W_synth.T if w[0] == h])
            W_t = np.array([[w[0], np.mean(w[1])] for w in W_t]).T
            W_t[0] = W_t[0].astype(int)
            #
            axes[0].plot(W_t[0].astype(int), W_t[1],'y-', alpha = 0.8)  # средняя по синетике
        axes[0].plot(W_predicted[0].astype(int), W_predicted[1],'k-', alpha = 1, label="pred")  # предсказанные
        axes[0].set_xlabel(parametr)
        axes[0].set_title(f'opps для предсказания/синтетики; масштаб: 100')
        axes[0].grid(True, linestyle="--", alpha=0.7)
        axes[0].legend()

        # ax = 1
        if len(W_synth)!=0:
            #W_synth = W_synth.T[:13].T
            axes[1].scatter(W_synth[0], W_synth[1], color='yellow', alpha = 0.7, s=10, label="synth")  # вся синтетика
            W_t = [[i, []] for i in (np.unique(W_synth[0]))]
            for j, h in enumerate(np.unique(W_synth[0])):
                W_t[j][1].extend([w[1] for w in W_synth.T if w[0] == h])
            W_t = np.array([[w[0], np.mean(w[1])] for w in W_t]).T
            W_t[0] = W_t[0].astype(int)
            #
            axes[1].plot(W_t[0].astype(int), W_t[1],'y-', alpha = 0.8)  # средняя по синетике
        if len(W_raw)!=0:
            axes[1].scatter(W_raw[0], W_raw[1], color='red', alpha = 0.7, s=10, label="raw")  # вся сырая дата
        
        axes[1].plot(W_predicted[0].astype(int), W_predicted[1],'k-', alpha = 1, label="pred")  # предсказанные
        axes[1].set_xlabel(parametr)
        axes[1].set_xlim(3,20)
        axes[1].set_title(f'opps для предсказания/синтетики/сырая; масштаб: 16')
        axes[1].grid(True, linestyle="--", alpha=0.7)
        axes[1].legend()

        graph_title = {key: {'first graph': fix_params[key]} for key in [x for x in list(doc.features_space.keys()) if x != parametr]}
        #graph_title.pop(parametr, None)
        print(f'params:\n{pd.DataFrame(graph_title).transpose()}')
        plt.ylabel("opps")
        plt.tight_layout()
        plt.show()

