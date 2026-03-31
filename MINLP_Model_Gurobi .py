import gurobipy as gp  # import the installed package
import gurobipy as grb
from gurobipy import GRB
from sklearn.metrics import adjusted_mutual_info_score
import pandas as pd
from sklearn.metrics import davies_bouldin_score,calinski_harabasz_score
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
import numpy as np
import datetime


#modified_data_ver2 data
def calculate_invalid_constraint(df_set):
    # Her kümede en az bir 'constraint' değeri 1 olan olup olmadığını kontrol et
    cluster_groups = df.groupby("assigned_cluster")["constraint"].sum()

    # En az bir 'constraint' = 1 içermeyen kümeler
    invalid_clusters = cluster_groups[cluster_groups == 0].index
    # Oran hesaplama
    total_clusters = df_set["assigned_cluster"].nunique()

    invalid_cluster_ratio = len(invalid_clusters) / total_clusters


    return invalid_cluster_ratio
def calculate_capacity_violation(df_set, cluster_capacity):

    # Her kümeye atanan nokta sayısını hesapla
    cluster_counts = df_set["assigned_cluster"].value_counts()

    # Kapasite ihlali yapan kümeleri bul
    violated_clusters = cluster_counts[cluster_counts > cluster_capacity].index
    print("violated_clusters", violated_clusters)
    # Oran hesaplama
    total_clusters = df_set["assigned_cluster"].nunique()
    print("total_clusters",total_clusters)

    violation_ratio = len(violated_clusters) / total_clusters
    print(violation_ratio)
    return violation_ratio
def dunn_index(X, labels):
    unique_labels = np.unique(labels)
    min_intercluster_dist = np.inf  # En küçük kümeler arası mesafe
    max_intracluster_dist = 0       # En büyük küme içi mesafe

    # Kümeler arasındaki en küçük mesafeyi bul
    for i in unique_labels:
        for j in unique_labels:
            if i != j:
                cluster_i = X[labels == i]
                cluster_j = X[labels == j]
                inter_dist = np.min(cdist(cluster_i, cluster_j))  # Küme merkezleri değil, noktalar arasındaki min mesafe
                min_intercluster_dist = min(min_intercluster_dist, inter_dist)

    # Kümeler içindeki en büyük mesafeyi bul
    for i in unique_labels:
        cluster_i = X[labels == i]
        intra_dist = np.max(cdist(cluster_i, cluster_i))  # Küme içindeki en büyük mesafe
        max_intracluster_dist = max(max_intracluster_dist, intra_dist)

    return min_intercluster_dist / max_intracluster_dist
timelimit = 36 #3600
datasets=['iris','heart']
percentages=[5,10,15,20]
gurobi_small_test_result=[]
gurobi_small_test_result.append(['Dataset','Percentage','nCluster','RunTime',"Adjusted Mutual Information (AMI)","Adjusted Rand Index (ARI)",'Davies-Bouldin','Calinski-Harabasz','Dunn','Silhoutte','invalid_constraint_rate','invalid_capacity_rate','GAP (%)','Model_Status'])

df_test=pd.read_excel('Real_ConsCluster_Test_Character.xlsx',skiprows=1)
df_test = df_test.iloc[:, 1:]

for Datasetname in datasets:
    for percentage in percentages:

        cluster_capacity = df_test.loc[df_test["Test Name"] == Datasetname, "cluster_capacity"].values[0]
        df = pd.read_csv(f"{Datasetname}_data_{percentage}.csv")

        true_labels = df['class']
        df.drop(columns=['class'], inplace=True)
        feature_constrate = df['constraint'].to_numpy()
        df.drop(['constraint'], axis=1, inplace=True)
        X = df.to_numpy()
        print(type(X))
        print(type(feature_constrate))
        k = true_labels.nunique()
        d = X.shape[1]
        n = X.shape[0]
        print(k, d, n)
        print(feature_constrate)
        print(cluster_capacity)

        model = grb.Model('k-means')

        X_min = X.min(axis=0)  # shape: (d,)
        X_max = X.max(axis=0)  # shape: (d,)

        # --- 2) lb/ub sözlüklerini hazırla (j,l) ile indeksli
        lb_mu = {(j, l): float(X_min[l]) for j in range(k) for l in range(d)}
        ub_mu = {(j, l): float(X_max[l]) for j in range(k) for l in range(d)}
        mu = model.addVars(range(k), range(d), lb=lb_mu, ub=ub_mu, name="mu")
        # mu = model.addVars(k, d, lb=-GRB.INFINITY, name='mu')  # küme merkezleri
        z = model.addVars(n, k, vtype=GRB.BINARY, name='z')  # gösterge değişkenleri

        # Kısıtlamalar
        for i in range(n):
            model.addConstr(grb.quicksum(z[i, j] for j in range(k)) == 1, name=f'data_point_assignment_{i}')

        for j in range(k):
            model.addConstr(grb.quicksum(z[i, j] for i in range(n)) <= cluster_capacity,
                            name=f"capacity_constraint_{j}")

        for j in range(k):
            model.addConstr(grb.quicksum(z[i, j] for i in range(n)) >= 1,
                            name=f"each_cluster_includes_at_least_one_{j}")
        # Her kümede en az bir araç sahibi olmalı
        for j in range(k):
            model.addConstr(grb.quicksum(z[i, j] * feature_constrate[i] for i in range(n)) >= 1,
                            name=f"vehicle_ownership_constraint_{j}")

        # Amaç fonksiyonu
        u = model.addVars(n, k, lb=-GRB.INFINITY, name="u")
        v = model.addVars(n, k, name="v")
        for j in range(k):
            for i in range(n):
                model.addConstr(u[i, j] == grb.quicksum((X[i, l] - mu[j, l]) * (X[i, l] - mu[j, l]) for l in range(d)))

        for j in range(k):
            for i in range(n):
                model.addConstr(v[i, j] == z[i, j] * u[i, j])

        obj = gp.quicksum(v[i, j] for i in range(n) for j in range(k))

        model.setObjective(obj, GRB.MINIMIZE)

        # Modeli çözme
        model.params.NonConvex = 2
        model.Params.TimeLimit = timelimit
        model.optimize()

        # Sonuçları yazdırma
        if model.Status == GRB.OPTIMAL:
            print("✅ Optimal çözüm bulundu.")
        elif model.Status == GRB.TIME_LIMIT:
            print("⏰ Zaman sınırına ulaşıldı.")
            print(f"Toplam süre: {model.Runtime:.2f} saniye")
            if model.SolCount > 0:
                print(f"En iyi mevcut amaç değeri: {model.ObjVal:.4f}")
                print(f"En iyi bound: {model.ObjBound:.4f}")
                print(f"Gap: {100 * model.MIPGap:.2f}%")
            else:
                print("Zaman sınırında geçerli bir çözüm bulunamadı.")
        else:
            print(f"⚠️ Optimizasyon tamamlanamadı. Status kodu: {model.Status}")

        # Eğer en az bir çözüm varsa (OPTIMAL veya TIME_LIMIT + incumbent)
        if model.SolCount > 0:
            print("\nCluster centers:")
            for j in range(k):
                print(f"  Cluster center {j}: {[mu[j, l].X for l in range(d)]}")

            print("\nData point assignments:")
            for i in range(n):
                for j in range(k):
                    if z[i, j].X > 0.5:
                        print(f"  Data point {i} assigned to cluster {j}")
        assigned_clusters = []
        for i in range(n):
            assigned_cluster = -1  # varsayılan değer
            for j in range(k):
                if z[i, j].X > 0.5:
                    assigned_cluster = j
                    break
            assigned_clusters.append(assigned_cluster)
        if model.SolCount > 0:
            gap_value = model.MIPGap * 100  # yüzde olarak
        else:
            gap_value = None
        print('Data point %d assigned to cluster %d' % (i, assigned_cluster))

        df['assigned_cluster'] = assigned_clusters
        df['constraint'] = feature_constrate
        df['class'] = true_labels
        df.to_excel("assinged.xlsx")

        db_score = davies_bouldin_score(df.drop(['assigned_cluster'], axis=1), df['assigned_cluster'])
        ch_score = calinski_harabasz_score(df.drop(['assigned_cluster'], axis=1), df['assigned_cluster'])
        dunn_score = dunn_index(df.drop(['assigned_cluster'], axis=1), df['assigned_cluster'])
        silhouette1 = silhouette_score(df.drop(['assigned_cluster'], axis=1), df['assigned_cluster'])
        invalid_constraint = calculate_invalid_constraint(df)
        invalid_capacity = calculate_capacity_violation(df, cluster_capacity)
        ami_score = adjusted_mutual_info_score(df['class'], df['assigned_cluster'])
        ari_score = adjusted_rand_score(df['class'], df['assigned_cluster'])
        gurobi_small_test_result.append(
            [Datasetname, percentage, k, model.Runtime, ami_score, ari_score, db_score, ch_score, dunn_score,
             silhouette1, invalid_constraint, invalid_capacity, gap_value, model.status])

# Tarih ve saat damgası oluştur (Örn: 20231027_1430)
timestampt = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# DataFrame oluşturma ve Excel'e kaydetme
gurobi_small_test_result_df = pd.DataFrame(gurobi_small_test_result)
gurobi_small_test_result_df.to_excel(f"gurobi_small_test_result_{timestampt}.xlsx")