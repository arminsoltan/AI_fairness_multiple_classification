import pandas as pd
import os
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from EqualizedOddPostProcessing import EqualizedOddsPostProcessing
from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score
import numpy as np
import random
import scipy as sp
import seaborn as sbn

from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

from balancers import BinaryBalancer, MulticlassBalancer
import tools


def make_dataset(df, favorable_label, unfavorable_label):
    # Create a new DataFrame with binary labels
    binary_labels_dataframe = pd.DataFrame({
        "predictions": (df["stage_pred"] == favorable_label).astype(int),
        "ground_truth": (df["stage"] == favorable_label).astype(int),
        "prediction_0": (df["prediction_0"]).astype(float),
        "prediction_1": (df["prediction_1"]).astype(float),
        "prediction_2": (df["prediction_2"]).astype(float),
        "prediction_3": (df["prediction_3"]).astype(float),
        "prediction_4": (df["prediction_4"]).astype(float),
        "race": df["race"]
    })
    return binary_labels_dataframe


def balance():
    df = pd.read_csv('final_train.csv')
    df.loc[df["race"] == 8, "race"] = 6
    df.loc[df["race"] == 9, "race"] = 7
    df.loc[df["race"] == 1, "race"] = 0
    df.loc[df["race"] == 2, "race"] = 1
    df.loc[df["race"] == 3, "race"] = 2
    df.loc[df["race"] == 4, "race"] = 3
    df.loc[df["race"] == 5, "race"] = 4
    df.loc[df["race"] == 6, "race"] = 5
    df.loc[df["race"] == 7, "race"] = 6
    races = [0, 1, 2, 3, 4, 5, 6]
    stages = [0, 1, 2, 3, 4]
    pred_stages = [0, 1, 2, 3, 4]
    dic = {"biopsy_id": [], "stage": [], "race": [], "prediction_0": [], "prediction_1": [], "prediction_2": [], "prediction_3": [], "prediction_4": [], "stage_pred": []}
    biopsy_id = "ff8c3857-722a-4c2c-98a6-52cc53971d42"
    value = 0.00001
    for race in races:
        for stage in stages:
            for pred_stage in pred_stages:
                dic["biopsy_id"].append(biopsy_id)
                dic["stage"].append(stage)
                dic["prediction_0"].append(value)
                dic["prediction_1"].append(value)
                dic["prediction_2"].append(value)
                dic["prediction_3"].append(value)
                dic["prediction_4"].append(value)
                dic["stage_pred"].append(pred_stage)
                dic["race"].append(race)
                value += 0.0001

    df1 = pd.DataFrame(dic)
    df = pd.concat([df, df1], axis=0)
    condition = df["race"].isin([0, 1, 2, 3, 4])
    print(condition)
    df2 = df.reset_index(drop=True)
    df2 = df[condition]
    a = df2.race.values
    y = df2.stage.values
    y_ = df2.stage_pred.values
    stats = tools.clf_metrics(y, y_)
    pb = MulticlassBalancer(y=y, y_=y_, a=a)
    pb.adjust_new(goal="demographic_parity", summary=False)
    pb.summary()


def bootstrap(df):
    n_bootstraps = 50
    bootstrap_samples = []

    for _ in range(n_bootstraps):
        # Sample with replacement
        sample_df = df.sample(n=len(df) // 2, replace=True)
        # Compute statistic - here we use the mean of 'Column1' as an example
        stat = sample_df['Column1'].mean()
        bootstrap_samples.append(stat)

    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_std = np.std(bootstrap_samples)
    # For a 95% confidence interval
    lower_bound = np.percentile(bootstrap_samples, 2.5)
    upper_bound = np.percentile(bootstrap_samples, 97.5)

    print(f"Bootstrap Mean: {bootstrap_mean}")
    print(f"Bootstrap Std: {bootstrap_std}")
    print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")


def show_information(i):
    mp = {
        0: "Non White fpr",
        1: "Non White tpr",
        2: "White fpr",
        3: "White tpr"
    }
    return mp[i]
def main():
    # Load your dataset
    random.seed(42)
    files = os.listdir("predictions/2/")
    for file in files:
        try:
            print("########################")
            print(file)
            df = pd.read_csv("predictions/3/{}".format(file))
            df["race"] = df["race"].replace([2, 3, 4, 5, 8, 9], "Non-White")
            df["race"] = df["race"].replace(["grouped_race"], "Non-White")
            df["race"] = df["race"].replace([1, "1"], "White")

            n_bootstraps = 50
            bootstrap_samples = [[], [], [], []]

            for _ in range(n_bootstraps):
                # Sample with replacement
                sample_df = df.sample(n=len(df) // 2, replace=True, random_state=random.seed(42))
                a = sample_df.race.values
                y = sample_df.actual_stage.values
                y_ = sample_df.pred_stage.values

                stats = tools.clf_metrics(y, y_)
                pb = BinaryBalancer(y=y, y_=y_, a=a)
                pb.adjust(goal="odds", summary=True)
                df_eval = pb.summary(org=True, adj=True)
                print(df_eval.values)

                bootstrap_samples[0].append(df_eval.values[0][1]) # Fpr Non-White
                bootstrap_samples[1].append(df_eval.values[0][2]) # TPR Non-White
                bootstrap_samples[2].append(df_eval.values[1][1]) # Fpr White
                bootstrap_samples[3].append(df_eval.values[1][2]) # TPR White

            for i in range(4):
                print(show_information(i))
                bootstrap_mean = np.mean(bootstrap_samples[i])
                bootstrap_std = np.std(bootstrap_samples[i])
                # For a 95% confidence interval
                lower_bound = np.percentile(bootstrap_samples[i], 2.5)
                upper_bound = np.percentile(bootstrap_samples[i], 97.5)

                print(f"Bootstrap Mean: {bootstrap_mean}")
                print(f"Bootstrap Std: {bootstrap_std}")
                print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]\n")

        except Exception as err:
            raise err

def multiple_labels():
    # Load your dataset
    random.seed(42)
    print("hello world")
    files = os.listdir("predictions/3/")
    for file in files:
        try:
            print("########################")
            print(file)
            df = pd.read_csv("predictions/3/{}".format(file))
            df["race"] = df["race"].replace([2, 3, 4, 5, 8, 9], "Non-White")
            df["race"] = df["race"].replace(["grouped_race"], "Non-White")
            df["race"] = df["race"].replace([1, "1"], "White")

            n_bootstraps = 50
            bootstrap_samples = [[], [], [], [], [], []]

            for _ in range(n_bootstraps):
                # Sample with replacement
                sample_df = df.sample(n=len(df) // 2, replace=True, random_state=42)
                a = sample_df.race.values
                y = sample_df.actual_stage.values
                y_ = sample_df.pred_stage.values

                stats = tools.clf_metrics(y, y_)
                pb = MulticlassBalancer(y=y, y_=y_, a=a)
                pb.adjust(goal="odds", summary=True, seed=random.seed(42))
                df_eval = pb.summary(org=True, adj=True)
                print(df_eval.values)

                bootstrap_samples[0].append(df_eval.values[0][0])  # Fpr 0
                bootstrap_samples[1].append(df_eval.values[0][1])  # TPR 0
                bootstrap_samples[2].append(df_eval.values[1][0])  # Fpr 1
                bootstrap_samples[3].append(df_eval.values[1][1])  # TPR 1
                bootstrap_samples[4].append(df_eval.values[2][0])  # FPR 2
                bootstrap_samples[5].append(df_eval.values[2][1])  # TPR 2

            for i in range(6):
                # print(show_information(i))
                bootstrap_mean = np.mean(bootstrap_samples[i])
                bootstrap_std = np.std(bootstrap_samples[i])
                # For a 95% confidence interval
                lower_bound = np.percentile(bootstrap_samples[i], 2.5)
                upper_bound = np.percentile(bootstrap_samples[i], 97.5)

                print(f"Bootstrap Mean: {bootstrap_mean}")
                print(f"Bootstrap Std: {bootstrap_std}")
                print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]\n")

        except Exception as err:
            print(err)


if __name__ == "__main__":
    # main()
    multiple_labels()
    # balance()