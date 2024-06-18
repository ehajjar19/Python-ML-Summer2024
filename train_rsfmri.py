import json
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV, Lasso, ElasticNetCV, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from plot_util import *
# from plot_util import plot_scatter, plot_loss, plot_all_metrics, plot_correlations, plot_corr_his


class BaseModel:
    def __init__(self, lambdas, K):
        self.lambdas = lambdas
        self.K = K
        self.model_eval = None

    def search_cv(self, x, y):
        raise NotImplementedError("Subclasses should implement this method.")

    def save_json(self, model, filename): #TODO make this subclass specific. Right now this won't work for MLP.
        model_attributes = {
            'alpha': model.alpha_,
            'l1_ratio': model.l1_ratio_ if hasattr(model, 'l1_ratio_') else 1, #Only for elasticnet
            'coef': model.coef_.tolist(),
            'intercept': model.intercept_,
            'mse_path': model.mse_path_.tolist(),
            'alphas': model.alphas_.tolist(),
            'dual_gap': model.dual_gap_,
            'n_iter': model.n_iter_,
            'n_features_in': model.n_features_in_,
            'feature_names_in': model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else []
        }
        filename = f"{filename}_{type(self).__name__}_details.json"
        with open(filename, 'w') as file:
            json.dump(model_attributes, file, indent=4)
        return filename

    def get_model_param(self, model): #TODO change to static?
        if getattr(model, "l1_ratio_", None):
            return {"alpha": model.alpha_, "l1_ratio": model.l1_ratio_}
        return {"alpha": model.alpha_}

    def evaluate_model(self, params, x, y):
        model = self.model_eval(**params)
        # model = self.model_class(alpha=params["alpha"], l1_ratio=params.get("l1_ratio", 1))
        cv = KFold(n_splits=self.K) # Consider adding shuffle=True if needed
        predictions = cross_val_predict(model, x, y, cv=cv)

        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        pearson_corr, p_value = pearsonr(y, predictions)

        model_stats = {**params,
                       "r2": r2, 
                       "mse": mse, 
                       "pearson_corr": pearson_corr, 
                       "p_value": p_value
                       }
        print(model_stats)
        return model_stats

    def get_predictions(self, params, x, y):
        model = self.model_eval(**params) #?????
        cv = KFold(n_splits=self.K) # Consider adding shuffle=True if needed
        predictions = cross_val_predict(model, x, y, cv=cv)
        return predictions
    
    @staticmethod
    def eval_pred(y, predictions):
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        pearson_corr, p_value = pearsonr(y, predictions)

        print(f"R^2 Score: {r2}")
        print(f"Pearson Correlation: {pearson_corr}")
        print(f"P-value: {p_value}")
        print(f"MSE: {mse}")
        return {"r2": r2, "mse": mse, "pearson_corr": pearson_corr, "p_value": p_value}

    @staticmethod
    def scatter_pred_y(y, predictions):
        plot_scatter(predictions, y)   


class LassoModel(BaseModel):
    def __init__(self, lambdas, K):
        super().__init__(lambdas, K)
        self.model_eval = Lasso  # Set the specific model class
    def search_cv(self, x, y):
        model = LassoCV(alphas=self.lambdas, cv=self.K).fit(x, y)
        return model


class ElasticNetModel(BaseModel):
    def __init__(self, lambdas, K):
        super().__init__(lambdas, K)
        self.model_eval = ElasticNet  # Set the specific model class

    def search_cv(self, x, y):
        model = ElasticNetCV(alphas=self.lambdas, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], cv=self.K).fit(x, y)
        return model

#TODO this model is WIP
class MLPModel(BaseModel):
    def __init__(self, lambdas, K):
        super().__init__(lambdas, K)
        self.model_eval = MLPRegressor

    def search_cv(self, x, y):
        model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', max_iter=200).fit(x,y) 
        #TODO Change to GridSearch CV
        #See train_mlp(subjects, x, label) in mlp_rsfmri_dti.py
        #TODO Write save_json method
        #TODO Write get_model_param(cv_result)
        #TODO make cross validation work. Currently no cross val.
        return model
    
    def get_model_param(self, model):
        return {"hidden_layer_sizes": (50,), "activation": 'relu', "max_iter": 200}
    

# TODO this model is WIP
class LassoMLPModel(BaseModel):
    def __init__(self, lambdas, K):
        super().__init__(lambdas, K)
        self.model_eval = MLPRegressor

    def search_cv(self, x, y):
        lasso = Lasso(alpha=0.1)
        lasso.fit(x, y)
        important_features = [i for i, coef in enumerate(lasso.coef_) if coef != 0]
        print(f"Important features: {important_features}")
        # Filter the dataset
        x_filtered = x[:, important_features]
        # Train MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', max_iter=200).fit(x_filtered, y)
        return model

    def get_model_param(self, model):
        return {"hidden_layer_sizes": (50,), "activation": 'relu', "max_iter": 200}

##################################################################

#TODO Remove this or move this inside the class
def get_model(model_type, lambdas, K):
    models = {
        "lasso": LassoModel,
        "elastic": ElasticNetModel,
        "mlp": MLPModel
        # Add more models here
    }
    return models.get(model_type)(lambdas, K)


def preprocess_data(subjects, x, label):
    '''
    Preprocess data: remove duplicate subjects in the data.
    '''
    sub = np.unique(subjects)
    idxValid = np.isin(subjects, sub)
    valid_data = x[idxValid, :]
    valid_label = label[idxValid]
    return valid_data, valid_label


#TODO Get rid of this function, and make set_lambda in the class.
def train_model(data, label, model_type='elastic', K=5, lambdas=[1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1], note=None):
    model = get_model(model_type, lambdas, K)
    cv_result = model.search_cv(data, label)
    model.save_json(cv_result, "ncanda_cv_raw/"+note) # Save cross validation result to a file.
    params = model.get_model_param(cv_result)

    # model.scatter_pred_y(label, model.get_predictions(params, data, label))
    #TODO save the prediction somewhere so that it can be accessed later. Instead of doing the above.

    return model.evaluate_model(params, data, label)


def train_all_region(model_type):
    '''
    For all DTI regions, train a model based on selected model_type and their respective decorrelated fMRI data.
    '''
    df = pd.read_csv('./diffusion/mori_ncanda_baseline_meet_criteria_fa_corrected_global_skeleton.csv')
    df = df.drop(columns=['arm', 'visit', 'subject'])

    result = []
    for column_name in df.columns:
        print("column name:", column_name)
        subjects = np.load(f"decorrelated/{column_name}_subjects.npy")
        x = np.load(f"decorrelated/{column_name}_x.npy")
        label = np.load(f"decorrelated/{column_name}_label.npy")

        valid_data, valid_label = preprocess_data(subjects, x, label)
        result_row = [column_name, *(train_model(valid_data, valid_label, model_type=model_type, note=column_name).values())]
        # TODO fix this so there's not .values(), and the result can be dynamically put into columns
        # TODO pass in a config file, so no need to hard code the lambda/ratio values.
        result.append(result_row)

        column_names = ["brain_region", "alpha", "l1_ratio", "r2_score", "mse", "pearson_correlation", "p-value"]
        df = pd.DataFrame(result, columns=column_names)
        df.to_csv(f'All_region_{model_type}.csv', index=False)


if __name__ == "__main__":
    train_all_region('elastic')  # Supported values include: 'lasso', 'mlp', 'elastic'. 

    plot_all_metrics('All_region_lasso.csv', 'lasso_metrics_plots.png')
    plot_all_metrics('All_region_elastic.csv', 'elastic_metrics_plots.png')

    # test_circlize()
    mat = plot_correlations('ncanda_cv_raw/AntCoronaRad_ElasticNetModel_details.json')
    plot_corr_his('ncanda_cv_raw/AntCoronaRad_ElasticNetModel_details.json')
    plot_sorted_line('ncanda_cv_raw/global_mori_ElasticNetModel_details.json')
    plot_circlize(mat)
    heatmap_circlize_folder()

    compare_coeff("global_mori_ElasticNetModel_details.json", "GenuCorpus_ElasticNetModel_details.json")
    batch_compare_coeff()
    output_fMRI_DTI_heatmap()
