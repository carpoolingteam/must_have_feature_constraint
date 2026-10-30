import os
import datetime
import numpy as np
import pandas as pd
import gurobipy as gp
import gurobipy as grb

from gurobipy import GRB
from scipy.spatial.distance import cdist

from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score
)


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def calculate_invalid_constraint(df_set):
    """
    Her kümede en az bir must-have gözlem bulunup bulunmadığını kontrol eder.
    Çıktı: must-have koşulunu ihlal eden kümelerin oranı.
    """
    cluster_groups = (
        df_set.groupby("assigned_cluster")["constraint"].sum()
    )

    invalid_clusters = cluster_groups[
        cluster_groups == 0
    ].index

    total_clusters = df_set["assigned_cluster"].nunique()

    if total_clusters == 0:
        return np.nan

    return len(invalid_clusters) / total_clusters


def calculate_capacity_violation(df_set, cluster_capacity):
    """
    Kapasiteyi aşan kümelerin oranını hesaplar.
    """
    cluster_counts = (
        df_set["assigned_cluster"].value_counts()
    )

    violated_clusters = cluster_counts[
        cluster_counts > cluster_capacity
    ].index

    total_clusters = df_set["assigned_cluster"].nunique()

    if total_clusters == 0:
        return np.nan

    return len(violated_clusters) / total_clusters


def dunn_index(X, labels):
    """
    Dunn index hesaplar.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    min_intercluster_dist = np.inf
    max_intracluster_dist = 0

    for i in unique_labels:
        for j in unique_labels:
            if i != j:
                cluster_i = X[labels == i]
                cluster_j = X[labels == j]

                inter_dist = np.min(
                    cdist(cluster_i, cluster_j)
                )

                min_intercluster_dist = min(
                    min_intercluster_dist,
                    inter_dist
                )

    for i in unique_labels:
        cluster_i = X[labels == i]

        intra_dist = np.max(
            cdist(cluster_i, cluster_i)
        )

        max_intracluster_dist = max(
            max_intracluster_dist,
            intra_dist
        )

    if max_intracluster_dist == 0:
        return np.nan

    return min_intercluster_dist / max_intracluster_dist


def get_status_name(status):
    """
    Gurobi status kodunu okunabilir metne çevirir.
    """
    status_dict = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT"
    }

    return status_dict.get(
        status,
        f"STATUS_{status}"
    )


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

timelimit = 3600
all_datasets=['appendicitis',
                'breast_cancer',
                'bupa',
                'ecoli',
                'glass',
                'haberman',
                'hayes-roth',
                'heart',
                'ionosphere',
                'iris',
                'led7digit',
                'monk-2',
                'movement_libras',
                'newthyroid',
                'saheart',
                'sonar',
                'soybean',
                'spectfheart',
                'tae',
                'vehicle',
                'zoo']
datasets = [
              'iris',
                'led7digit',
                'monk-2',
                'movement_libras',
                'newthyroid',
                'saheart',
                'sonar',
                'soybean',
                'spectfheart',
                'tae',
                'vehicle',
                'zoo'
]

percentages = [5, 10, 15, 20]

GUROBI_SEED = 42

os.makedirs(
    "gurobi_logs",
    exist_ok=True
)

os.makedirs(
    "gurobi_assignments",
    exist_ok=True
)


# ============================================================
# RESULT COLUMNS
# ============================================================

result_columns = [
    'Dataset',
    'Percentage',

    'n',
    'd',
    'nCluster',
    'Cluster_Capacity',
    'MustHave_Count',

    'SSE_Incumbent',
    'Best_Bound',
    'GAP_Percent',
    'RunTime_sec',
    'Time_Limit_sec',
    'Solver_Status',
    'Optimality_Proven',
    'Solution_Count',

    'Adjusted Mutual Information (AMI)',
    'Adjusted Rand Index (ARI)',
    'Davies-Bouldin',
    'Calinski-Harabasz',
    'Dunn',
    'Silhouette',

    'invalid_constraint_rate',
    'invalid_capacity_rate'
]

gurobi_results = []


# ============================================================
# READ TEST CHARACTERISTICS
# ============================================================

df_test = pd.read_excel(
    'Real_ConsCluster_Test_Character.xlsx',
    skiprows=1
)

df_test = df_test.iloc[:, 1:]


# ============================================================
# MAIN LOOP
# ============================================================

for Datasetname in datasets:

    for percentage in percentages:

        print("\n" + "=" * 80)
        print(
            f"Dataset: {Datasetname} | "
            f"Must-have percentage: {percentage}%"
        )
        print("=" * 80)

        cluster_capacity = df_test.loc[
            df_test["Test Name"] == Datasetname,
            "cluster_capacity"
        ].values[0]

        data = pd.read_csv(
            f"{Datasetname}_data_{percentage}.csv"
        )

        true_labels = data['class'].copy()

        feature_constraint = (
            data['constraint']
            .to_numpy()
            .astype(int)
        )

        feature_columns = [
            col for col in data.columns
            if col not in [
                'class',
                'constraint'
            ]
        ]

        X = data[
            feature_columns
        ].to_numpy(dtype=float)

        k = true_labels.nunique()
        n = X.shape[0]
        d = X.shape[1]

        must_have_count = int(
            feature_constraint.sum()
        )

        print(
            f"n = {n}, "
            f"d = {d}, "
            f"k = {k}, "
            f"capacity = {cluster_capacity}, "
            f"must-have count = {must_have_count}"
        )

        # ====================================================
        # FEASIBILITY PRE-CHECKS
        # ====================================================

        # 1) Her kümede en az bir must-have olması gerektiği için
        # must-have sayısı en az k olmalı
        if must_have_count < k:
            raise ValueError(
                f"Infeasible instance: "
                f"must-have count ({must_have_count}) < k ({k}) "
                f"for {Datasetname}, {percentage}%."
            )

        # 2) Toplam kapasite tüm gözlemleri karşılamalı
        if n > k * cluster_capacity:
            raise ValueError(
                f"Infeasible instance: "
                f"n ({n}) > total capacity "
                f"({k * cluster_capacity}) "
                f"for {Datasetname}, {percentage}%."
            )

        # 3) Her küme boş olamayacağı için
        # gözlem sayısı en az k olmalı
        if n < k:
            raise ValueError(
                f"Infeasible instance: "
                f"n ({n}) < k ({k}) "
                f"for {Datasetname}, {percentage}%."
            )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = grb.Model(
            f'CCC_MHF_{Datasetname}_{percentage}'
        )

        model.Params.NonConvex = 2
        model.Params.TimeLimit = timelimit
        model.Params.Seed = GUROBI_SEED

        log_file = os.path.join(
            "gurobi_logs",
            f"{Datasetname}_{percentage}.log"
        )

        model.Params.LogFile = log_file

        # ----------------------------------------------------
        # VARIABLES
        # ----------------------------------------------------

        X_min = X.min(axis=0)
        X_max = X.max(axis=0)

        lb_mu = {
            (j, l): float(X_min[l])
            for j in range(k)
            for l in range(d)
        }

        ub_mu = {
            (j, l): float(X_max[l])
            for j in range(k)
            for l in range(d)
        }

        mu = model.addVars(
            range(k),
            range(d),
            lb=lb_mu,
            ub=ub_mu,
            name="mu"
        )

        z = model.addVars(
            n,
            k,
            vtype=GRB.BINARY,
            name="z"
        )

        # ----------------------------------------------------
        # ASSIGNMENT CONSTRAINT
        # ----------------------------------------------------

        for i in range(n):

            model.addConstr(
                grb.quicksum(
                    z[i, j]
                    for j in range(k)
                ) == 1,
                name=f"assignment_{i}"
            )

        # ----------------------------------------------------
        # CAPACITY CONSTRAINT
        # ----------------------------------------------------

        for j in range(k):

            model.addConstr(
                grb.quicksum(
                    z[i, j]
                    for i in range(n)
                )
                <= cluster_capacity,
                name=f"capacity_{j}"
            )

        # ----------------------------------------------------
        # NON-EMPTY CLUSTER CONSTRAINT
        # ----------------------------------------------------

        for j in range(k):

            model.addConstr(
                grb.quicksum(
                    z[i, j]
                    for i in range(n)
                )
                >= 1,
                name=f"nonempty_{j}"
            )

        # ----------------------------------------------------
        # MUST-HAVE CONSTRAINT
        # ----------------------------------------------------

        for j in range(k):

            model.addConstr(
                grb.quicksum(
                    z[i, j] *
                    feature_constraint[i]
                    for i in range(n)
                )
                >= 1,
                name=f"must_have_{j}"
            )

        # ----------------------------------------------------
        # OBJECTIVE FUNCTION: WITHIN-CLUSTER SSE
        # ----------------------------------------------------

        u = model.addVars(
            n,
            k,
            lb=0,
            name="u"
        )

        v = model.addVars(
            n,
            k,
            lb=0,
            name="v"
        )

        for j in range(k):

            for i in range(n):

                model.addConstr(
                    u[i, j]
                    ==
                    grb.quicksum(
                        (
                            X[i, l] -
                            mu[j, l]
                        )
                        *
                        (
                            X[i, l] -
                            mu[j, l]
                        )
                        for l in range(d)
                    ),
                    name=f"squared_distance_{i}_{j}"
                )

        for j in range(k):

            for i in range(n):

                model.addConstr(
                    v[i, j]
                    ==
                    z[i, j] *
                    u[i, j],
                    name=f"selected_distance_{i}_{j}"
                )

        obj = gp.quicksum(
            v[i, j]
            for i in range(n)
            for j in range(k)
        )

        model.setObjective(
            obj,
            GRB.MINIMIZE
        )

        # ----------------------------------------------------
        # OPTIMIZE
        # ----------------------------------------------------

        model.optimize()

        # ----------------------------------------------------
        # SOLVER RESULTS
        # ----------------------------------------------------

        status_name = get_status_name(
            model.Status
        )

        runtime = model.Runtime
        solution_count = model.SolCount

        optimality_proven = (
            model.Status == GRB.OPTIMAL
        )

        if solution_count > 0:

            sse_value = model.ObjVal
            best_bound = model.ObjBound
            gap_value = model.MIPGap * 100

        else:

            sse_value = np.nan
            best_bound = np.nan
            gap_value = np.nan

        print("\nSolver summary")
        print("-" * 50)

        print(
            "Status:",
            status_name
        )

        print(
            f"Runtime: "
            f"{runtime:.2f} sec"
        )

        print(
            f"Time limit: "
            f"{timelimit} sec"
        )

        print(
            "Solution count:",
            solution_count
        )

        if solution_count > 0:

            print(
                f"Incumbent SSE: "
                f"{sse_value:.6f}"
            )

            print(
                f"Best bound: "
                f"{best_bound:.6f}"
            )

            print(
                f"MIP gap: "
                f"{gap_value:.4f}%"
            )

            print(
                "Optimality proven:",
                optimality_proven
            )

        # ----------------------------------------------------
        # IF FEASIBLE SOLUTION EXISTS
        # ----------------------------------------------------

        if solution_count > 0:

            assigned_clusters = []

            for i in range(n):

                assigned_cluster = -1

                for j in range(k):

                    if z[i, j].X > 0.5:

                        assigned_cluster = j
                        break

                assigned_clusters.append(
                    assigned_cluster
                )

            assigned_clusters = np.array(
                assigned_clusters
            )

            # Güvenlik kontrolü
            if np.any(
                assigned_clusters == -1
            ):
                raise RuntimeError(
                    f"Unassigned observation detected "
                    f"in {Datasetname}, {percentage}%."
                )

            # ------------------------------------------------
            # RESULT DATAFRAME
            # ------------------------------------------------

            result_df = pd.DataFrame(
                X,
                columns=feature_columns
            )

            result_df[
                'constraint'
            ] = feature_constraint

            result_df[
                'class'
            ] = true_labels.to_numpy()

            result_df[
                'assigned_cluster'
            ] = assigned_clusters

            assignment_file = os.path.join(
                "gurobi_assignments",
                f"assigned_"
                f"{Datasetname}_"
                f"{percentage}.xlsx"
            )

            result_df.to_excel(
                assignment_file,
                index=False
            )

            # ------------------------------------------------
            # CLUSTERING QUALITY METRICS
            # Sadece gerçek feature'lar = X
            # ------------------------------------------------

            db_score = (
                davies_bouldin_score(
                    X,
                    assigned_clusters
                )
            )

            ch_score = (
                calinski_harabasz_score(
                    X,
                    assigned_clusters
                )
            )

            dunn_score = (
                dunn_index(
                    X,
                    assigned_clusters
                )
            )

            silhouette_value = (
                silhouette_score(
                    X,
                    assigned_clusters
                )
            )

            ami_score = (
                adjusted_mutual_info_score(
                    true_labels,
                    assigned_clusters
                )
            )

            ari_score = (
                adjusted_rand_score(
                    true_labels,
                    assigned_clusters
                )
            )

            invalid_constraint = (
                calculate_invalid_constraint(
                    result_df
                )
            )

            invalid_capacity = (
                calculate_capacity_violation(
                    result_df,
                    cluster_capacity
                )
            )

        else:

            db_score = np.nan
            ch_score = np.nan
            dunn_score = np.nan
            silhouette_value = np.nan

            ami_score = np.nan
            ari_score = np.nan

            invalid_constraint = np.nan
            invalid_capacity = np.nan

        # ----------------------------------------------------
        # APPEND RESULTS
        # ----------------------------------------------------

        gurobi_results.append([
            Datasetname,
            percentage,

            n,
            d,
            k,
            cluster_capacity,
            must_have_count,

            sse_value,
            best_bound,
            gap_value,
            runtime,
            timelimit,
            status_name,
            optimality_proven,
            solution_count,

            ami_score,
            ari_score,
            db_score,
            ch_score,
            dunn_score,
            silhouette_value,

            invalid_constraint,
            invalid_capacity
        ])

        # ----------------------------------------------------
        # INTERMEDIATE SAVE
        # ----------------------------------------------------

        intermediate_df = pd.DataFrame(
            gurobi_results,
            columns=result_columns
        )

        intermediate_df.to_excel(
            "revision_gurobi_intermediate.xlsx",
            index=False
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

timestamp = (
    datetime.datetime.now()
    .strftime("%Y%m%d_%H%M")
)

gurobi_result_df = pd.DataFrame(
    gurobi_results,
    columns=result_columns
)

output_file = (
    f"revision_gurobi_"
    f"one_hour_results_"
    f"{timestamp}.xlsx"
)

gurobi_result_df.to_excel(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("ALL GUROBI EXPERIMENTS COMPLETED")
print("=" * 80)

print(
    "Final results saved to:",
    output_file
)