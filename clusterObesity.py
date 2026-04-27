import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from kneed import KneeLocator
from matplotlib import pyplot as plt


cols_binarias = [
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "SCC",
]
cols_ordinais = ["CAEC", "CALC", "NObeyesdad"]
cols_nominais = ["Gender", "MTRANS"]


def preprocess(df):
    df_num = df.select_dtypes(include=np.number)
    df_bin = df[cols_binarias]
    df_ord = df[cols_ordinais]
    df_nom = df[cols_nominais]

    scaler = MinMaxScaler()
    arr_numerical = scaler.fit_transform(df_num)
    df_num_scaled = pd.DataFrame(
        arr_numerical,
        columns=df_num.columns,
        index=df.index
    )

    df_bin_encoded = df_bin.copy()
    # professor aqui eu achei que é melhor fazer assim, 
    # porque essas colunas de fato tem que ter um certo peso quando for 1.
    for col in cols_binarias:
        df_bin_encoded[col] = df_bin_encoded[col].map({"yes": 1, "no": 0})

    ordinal_encoder = OrdinalEncoder(categories=[
        ["no", "Sometimes", "Frequently", "Always"],
        ["no", "Sometimes", "Frequently", "Always"],
        ["Insufficient_Weight","Normal_Weight","Overweight_Level_I","Overweight_Level_II","Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"],
    ])
    array_ordinal = ordinal_encoder.fit_transform(df_ord)
    df_ord_encoded = pd.DataFrame(array_ordinal, columns=cols_ordinais, index=df.index)

    # professor o drop first n é nada haver com dummy trap
    # é só pq  a coluna genero estava muito influente
    # aí isso diminui o peso dela na curva
    ohe_nom = OneHotEncoder(drop='first',sparse_output=False, dtype=np.int8)
    arr_nom = ohe_nom.fit_transform(df_nom)
    cols_nom = ohe_nom.get_feature_names_out(cols_nominais)
    df_nom_encoded = pd.DataFrame(arr_nom, columns=cols_nom, index=df.index)

    df_final = pd.concat(
        [df_num_scaled, df_bin_encoded, df_ord_encoded, df_nom_encoded],
        axis=1,
    )

    return df_final, scaler, ordinal_encoder, ohe_nom


def find_clusters_number(df_final):
    inertias = []
    for k in range(1, 70):
        model = KMeans(n_clusters=k, n_init=10, random_state=1)
        model.fit(df_final)
        inertias.append(model.inertia_)

    knee = KneeLocator(range(1, 70), inertias, curve="convex", direction="decreasing")
    clusters_num = knee.knee

    plt.plot(range(1, 70), inertias, marker="o")
    plt.title("Elbow graphic for Obesity dataset")
    plt.xlabel("K (number of Clusters)")
    plt.ylabel("Inertia")
    plt.scatter(clusters_num, inertias[clusters_num], color='red', s=100, zorder=5)
    plt.savefig("obesity-elbow.png")
    plt.close()

    return clusters_num


def classify_instance(instance, model, scaler, ordinal_encoder, ohe_nom, df_columns):
    instance_df = pd.DataFrame([instance])

    num_cols = [c for c in instance_df.columns if c not in cols_binarias + cols_ordinais + cols_nominais]
    arr_num = scaler.transform(instance_df[num_cols])
    df_num_scaled = pd.DataFrame(arr_num, columns=num_cols)

    df_bin = instance_df[cols_binarias].copy()
    for col in cols_binarias:
        df_bin[col] = df_bin[col].map({"yes": 1, "no": 0})

    arr_ord = ordinal_encoder.transform(instance_df[cols_ordinais])
    df_ord = pd.DataFrame(arr_ord, columns=cols_ordinais)

    arr_nom = ohe_nom.transform(instance_df[cols_nominais])
    cols_nom = ohe_nom.get_feature_names_out(cols_nominais)
    df_nom = pd.DataFrame(arr_nom, columns=cols_nom)

    instance_final = pd.concat([df_num_scaled, df_bin, df_ord, df_nom], axis=1)
    instance_final = instance_final.reindex(columns=df_columns, fill_value=0)

    cluster = model.predict(instance_final)[0]
    return cluster


def main():
    df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

    df_final, scaler, ordinal_encoder, ohe_nom = preprocess(df)

    clusters_num = find_clusters_number(df_final)

    model = KMeans(n_clusters=clusters_num, n_init=10, random_state=1)
    model.fit(df_final)

    df_final['cluster'] = model.labels_
    # professor isso vai ser o mesmo que pegar dos centroides 
    # pois o cluster tem esses valores por si só
    print(df_final.groupby('cluster').mean().T)

    instance = {
        "Age": 25, "Height": 1.75, "Weight": 80,
        "FCVC": 2, "NCP": 3, "CH2O": 2, "FAF": 1, "TUE": 1,
        "family_history_with_overweight": "yes",
        "FAVC": "yes", "SMOKE": "no", "SCC": "no",
        "CAEC": "Sometimes", "CALC": "no",
        "NObeyesdad": "Normal_Weight",
        "Gender": "Male", "MTRANS": "Public_Transportation",
    }

    feature_cols = [c for c in df_final.columns if c != 'cluster']
    cluster = classify_instance(instance, model, scaler, ordinal_encoder, ohe_nom, feature_cols)
    print(f"\nNova instância classificada no cluster: {cluster}")


if __name__ == "__main__":
    main()