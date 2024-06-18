import os, sys
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import zscore


def decorrelate(column):
    """
    Using MixedLM to decorrelate confounding variables(race, age, etc.) in imaging data.
    """
    # Load variables
    demo = pd.read_csv('./ncanda_redcap/demographics.csv')
    demo = demo[demo['arm'] == 'standard'] # TODO What is NCanda standard arm/recovery arm?
    clinical = pd.read_csv('./ncanda_redcap/clinical.csv')
    cahalan = pd.read_csv('./resources/cahalan_plus_drugs.csv')
    mori = pd.read_csv('./diffusion/mori_ncanda_baseline_meet_criteria_fa_corrected_global_skeleton.csv')

    # Load rs-fMRI correlation matrices and DTI measures
    idx = np.ones((180, 180)) #180 correspond to the 180 columns of rs-fMRI
    idx = np.triu(idx, 1) #All diagonal element is 1 so we remove them in this mask as well.

    x = []
    label = []
    subjects = []
    age = []
    sex = []
    race = []
    ses = []
    scanner = []
    sub_idx = []
    alcohol = []
    used = np.zeros(len(mori), dtype=bool) # indicate if that sample entry is used.
    # Mori is the DTI (diffusion tensor imaging) data, stored in the /diffusion folder.

    for i in tqdm(range(len(mori)), desc="Filtering and combining input data."):
        rsfmri_filename = f"./rsfmri_correlation_matrix/rsfmri_correlation_matrix/{mori['subject'][i]}_{mori['visit'][i]}.mat"
        cahalan_sub = cahalan[(cahalan['subject'] == mori['subject'][i]) & (cahalan['visit'] == mori['visit'][i])]
        demo_sub = demo[(demo['subject'] == mori['subject'][i]) & (demo['visit'] == mori['visit'][i])]
        demo_sub_baseline = demo[(demo['subject'] == mori['subject'][i]) & (demo['visit'] == 'baseline')]
        clinical_sub_baseline = clinical[(clinical['subject'] == mori['subject'][i]) & (clinical['visit'] == 'baseline')]
        ses_sub = clinical_sub_baseline['ses_parent_yoe'].values[0] if not clinical_sub_baseline.empty else np.nan # nan is handled later in the code.
        race_sub = [demo_sub_baseline['race_label'].values[0] == label for label in ['Caucasian/White', 'African-American/Black', 'Asian']] if not demo_sub_baseline.empty else [np.nan, np.nan, np.nan]

        if os.path.exists(rsfmri_filename) and demo_sub['visit_age'].values[0] <= 21: #Limit visit_age < 21
            used[i] = True
            mat_data = scipy.io.loadmat(rsfmri_filename)
            T = mat_data['T']
            # idx = mat_data['idx']
            x.append(T[idx > 0.5].tolist()) # idx > 0.5 creates a true false matrix, in this case it is upper triangular.
            # this selects all elements of T that are above the main diagonal, then append to x.
            label.append(mori[column][i])
            subjects.append(mori['subject'][i])
            age.append(demo_sub['visit_age'].values[0])
            sex.append(demo_sub['sex'].values[0] == 'F')
            ses.append(ses_sub) #parent years of education
            race.append(race_sub)
            scanner.append(demo_sub['scanner'].values[0] == 'ge')
            sub_idx.append(int(mori['subject'][i][8:13])) #Adding the subject id

            if cahalan_sub.empty:
                alcohol.append([np.nan, np.nan, np.nan])
            else:
                alcohol.append(cahalan_sub[['frequency', 'drink_days_past_year', 'max_drinks_past_year']].values[0])

    x = np.array(x)
    label = np.array(label)
    subjects = np.array(subjects)
    age = np.array(age)
    sex = np.array(sex)
    ses = np.array(ses)
    race = np.array(race)
    scanner = np.array(scanner)
    sub_idx = np.array(sub_idx)
    alcohol = np.array(alcohol)
    # Replace NaNs in ses with the mean of ses
    ses[np.isnan(ses)] = np.nanmean(ses)

    # Confounding effects removal (age, sex, scanner type, socioeconomic status, race)
    covariate = np.column_stack((age, sex.astype(int), scanner.astype(int), ses, race))

    for i in tqdm(range(x.shape[1]), desc="Confounding effects removal for each column, this takes a while."):
        data = pd.DataFrame({
            'age': age,
            'sex': sex,
            'scanner': scanner,
            'ses': ses,
            'race1': race[:, 0],
            'race2': race[:, 1],
            'race3': race[:, 2],
            'groups': sub_idx,
            'feature': x[:, i]
        })
        lme = MixedLM.from_formula('feature ~ age + sex + scanner + ses + race1 + race2 + race3 + (1|groups)', groups="groups", data=data)
        result = lme.fit()
        beta = result.params[1:-2]  # Coefficients except for the intercept (changed by Robby to delete last two elements)
        # Removed the last two elements in result.params because the last two are 1|groups and groups var.
        x[:, i] = zscore(x[:, i] - np.dot(covariate, beta))

    # Decorrelate the label:
    data_with_label = pd.DataFrame({
        'age': age,
        'sex': sex,
        'scanner': scanner,
        'ses': ses,
        'race1': race[:, 0],
        'race2': race[:, 1],
        'race3': race[:, 2],
        'sub_idx': sub_idx,
        'feature': label
    })
    lme_label = MixedLM.from_formula('feature ~ age + sex + scanner + ses + race1 + race2 + race3 + (1|sub_idx)', groups="sub_idx", data=data_with_label)
    result_label = lme_label.fit()
    beta_label = result_label.params[1:-2]

    label = zscore(label - np.dot(covariate, beta_label))

    try:
        np.save(f"decorrelated/{column}_subjects.npy", subjects)
        np.save(f"decorrelated/{column}_x.npy", x)
        np.save(f"decorrelated/{column}_label.npy", label)
        # If you want to store the linear predictor of the fixed effects:
        # linear_predictor = np.dot(covariate, beta_label)
        # np.save("linear_predictor.npy", linear_predictor)
    except:
        print("Error when saving files.")


if __name__ == "__main__":
    print(sys.path)
    sys.path.append(r"C:\Users\Eddie\OneDrive\ML Research 2024\code_data_rsfmri_dti_ncanda\code_data_rsfmri_dti_ncanda")
    print(sys.path)

    print(os.path.exists("diffusion"));
    file = r"C:\Users\Eddie\OneDrive\ML Research 2024\rsfmri_dti-main\ncanda_redcap_dict"


    columns = ["global_mori","MidCblmPed","PontineCros","GenuCorpus","BodyCorpus","SplnCorpus","Fornix","CortspnlTrct","MedLemniscus","InfCblmPed",
               "SupCblmPed","CerebralPed","AntIntCap","PosIntCap","RetroIntCap","AntCoronaRad","SupCoronaRad","PosCoronaRad","PosThalamRad",
               "SagStratum","ExternalCap","CingAntMid","CingInf","StriaTerminali","SupLongFasc","SupFrntOccFasc","UncinateFac","Tapetum"]

    #You can either pass in argument with -c flag to specify columns, or just change the "columns" variable above.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--column", help="The dti column to use as label", type=str)
    args = parser.parse_args()
    if args.column:
        columns = args.column.split(',')

    print("Decorrelating for these columns:\n", columns)

    for column in columns:
        decorrelate(column)
