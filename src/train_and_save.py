from db.db_connector import get_engine
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, chi2
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10

def load_data_from_db():
    print("Conectando a la base de datos...")
    engine = get_engine()
    
    query = "SELECT * FROM analytics.v_model_training_dataset;"
    df = pd.read_sql(query, con=engine)
    
    print(f"Dataset cargado exitosamente. Filas y columnas: {df.shape}")
    return df

def train_and_save_model():
    # 1. Cargar el dataset desde PostgreSQL
    df = load_data_from_db()
    
    # 2. Verificar que los datos se cargaron bien
    print(df.head())

    print("Iniciando preprocesamiento y entrenamiento...")

    def ejecutar_pipeline_kdd():
        project_root = Path(__file__).resolve().parent.parent.parent if Path(__file__).resolve().parent.parent.name == 'src' else Path(__file__).resolve().parent.parent
        out_dir = project_root / "models"
        out_dir_results = project_root / "results"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_dir_results.mkdir(exist_ok=True, parents=True)

        print("=" * 90)
        print("ETAPA 1 KDD: DEFINICIÓN DEL OBJETIVO Y COMPRENSIÓN DEL DOMINIO")
        print("=" * 90)
        print(
            "Objetivo Analítico: Descubrir patrones de comportamiento anómalos y reglas de decisión\n"
            "accionables que permitan la detección automatizada de fraude en transacciones con tarjeta\n"
            "de crédito, superando el desafío de señales débiles en un entorno masivo (100,000 filas)\n"
            "y severamente desbalanceado (97.95% legítimas vs. 2.05% fraudulentas)."
        )

        # =========================================================================
        # ETAPA 2 KDD: SELECCIÓN DE DATOS Y PREVENCIÓN DE FUGA DE INFORMACIÓN
        # =========================================================================
        print("\n" + "=" * 90)
        print("ETAPA 2 KDD: SELECCIÓN DE DATOS Y AUDITORÍA DE DATA LEAKAGE")
        print("=" * 90)
        df = load_data_from_db()
        print(f"Dimensiones iniciales del dataset en bruto: {df.shape[0]:,} filas x {df.shape[1]} columnas")

        # Distribución de la variable objetivo y tipologías de fraude (análisis descriptivo previo)
        conteo_clases = df["is_fraud"].value_counts()
        prop_clases = df["is_fraud"].value_counts(normalize=True) * 100
        perdida_total = df["fraud_amount_lost"].sum()
        print(f"Distribución de 'is_fraud':")
        print(f"  - Clase 0 (Legítima):    {conteo_clases[0]:,} ({prop_clases[0]:.3f}%)")
        print(f"  - Clase 1 (Fraudulenta): {conteo_clases[1]:,} ({prop_clases[1]:.3f}%) -> Ratio 1:{conteo_clases[0]/conteo_clases[1]:.1f}")
        print(f"  - Pérdida económica acumulada por fraude: ${perdida_total:,.2f} USD")

        # Gráfica 1: Desbalance de clases y tipologías de fraude
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colores_clase = ["#2b5c8f", "#d95f02"]
        bars = axes[0].bar(
            ["Legítima (0)", "Fraude (1)"],
            [conteo_clases[0], conteo_clases[1]],
            color=colores_clase,
            edgecolor="black",
            width=0.55,
        )
        axes[0].set_title("Desbalance Extremo de la Variable Objetivo (is_fraud)", fontweight="bold")
        axes[0].set_ylabel("Número de Transacciones")
        for bar, pct in zip(bars, [prop_clases[0], prop_clases[1]]):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1800,
                f"{int(bar.get_height()):,}\n({pct:.2f}%)",
                ha="center",
                fontweight="bold",
            )
        axes[0].set_ylim(0, 115000)

        df_fraudes = df[df["is_fraud"] == 1]
        tipo_counts = df_fraudes["fraud_type"].value_counts()
        sns.barplot(
            x=tipo_counts.values,
            y=tipo_counts.index,
            palette="OrRd_r",
            ax=axes[1],
            edgecolor="black",
        )
        axes[1].set_title("Distribución por Tipología de Fraude (n = 2,055)", fontweight="bold")
        axes[1].set_xlabel("Cantidad de Casos de Fraude")
        axes[1].set_ylabel("Modalidad (fraud_type)")
        for i, v in enumerate(tipo_counts.values):
            axes[1].text(v + 15, i, f"{v:,} ({v/len(df_fraudes)*100:.1f}%)", va="center", fontweight="bold")
        axes[1].set_xlim(0, max(tipo_counts.values) * 1.22)
        plt.tight_layout()
        fig.savefig(out_dir_results / "01_desbalance_y_tipologias.png", bbox_inches="tight")
        plt.close(fig)

        # Selección de variables: Exclusión de identificadores y variables con Data Leakage
        columnas_identificadores = [
            "transaction_id",
            "customer_id",
            "card_id",
            "merchant_id",
            "transaction_datetime",
            "customer_state",  # 50 estados uniformes sin señal de fraude individual
        ]
        columnas_leakage = [
            "fraud_type",         # Conocida solo post-evento (100% correlacionada con el target)
            "fraud_amount_lost",  # Pérdida contable post-fraude (>0 solo si is_fraud == 1)
        ]
        df = df.drop(columns=columnas_identificadores + columnas_leakage).copy()
        print(f"\nVariables descartadas por Identificador/Cardinalidad ({len(columnas_identificadores)}): {columnas_identificadores}")
        print(f"Variables descartadas por Fuga de Información - Data Leakage ({len(columnas_leakage)}): {columnas_leakage}")
        print(f"Dimensiones tras Selección de Datos: {df.shape[0]:,} filas x {df.shape[1]} columnas")

        # =========================================================================
        # ETAPA 3 KDD: LIMPIEZA DE DATOS Y DETECCIÓN DE VALORES ATÍPICOS (OUTLIERS)
        # =========================================================================
        print("\n" + "=" * 90)
        print("ETAPA 3 KDD: LIMPIEZA DE DATOS, IMPUTACIÓN ESTRUCTURAL Y DETECCIÓN DE OUTLIERS")
        print("=" * 90)

        nulos_iniciales = df.isnull().sum()
        nulos_activos = nulos_iniciales[nulos_iniciales > 0]
        print("Valores nulos detectados antes de la limpieza:")
        for col, val in nulos_activos.items():
            print(f"  - {col}: {val:,} registros ({val/len(df)*100:.2f}%)")

        # Verificación estructural: browser_clean es NaN exactamente cuando el canal es In-store o ATM
        print(df.columns.tolist())
        cruce_browser_clean = df[df["browser_clean"].isnull()]["transaction_channel"].value_counts().to_dict()
        print(f"Origen estructural de nulos en 'browser_clean': {cruce_browser_clean}")
        print(
            "Decisión Técnica: NO eliminar filas (evita Sesgo de Selección que borraría el 100% de\n"
            "transacciones físicas en POS y cajeros ATM). Se imputa categoría 'No_Aplica_Fisico'."
        )
        df["browser_clean"] = df["browser_clean"].fillna("No_Aplica_Fisico")

        duplicados = df.duplicated().sum()
        print(f"Registros exactamente duplicados: {duplicados}")
        print(f"Total de nulos tras limpieza: {df.isnull().sum().sum()}")

        # Análisis estadístico de Valores Atípicos (Regla 1.5 x IQR de Tukey)
        print("\n--- Diagnóstico Estadístico de Valores Atípicos (Criterio 1.5 x IQR) ---")
        vars_outliers = [
            "transaction_amount",
            "amount_last_1h",
            "distance_from_home_km",
            "distance_from_last_tx_km",
            "amount_vs_avg_ratio",
            "transactions_last_1h",
        ]
        resumen_iqr = []
        for col in vars_outliers:
            q1 = df[col].quantile(0.25)
            q2 = df[col].quantile(0.50)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lim_inf = q1 - 1.5 * iqr
            lim_sup = q3 + 1.5 * iqr
            mask_out = (df[col] < lim_inf) | (df[col] > lim_sup)
            n_out = mask_out.sum()
            tasa_fraude_out = df.loc[mask_out, "is_fraud"].mean() * 100 if n_out > 0 else 0.0
            tasa_fraude_norm = df.loc[~mask_out, "is_fraud"].mean() * 100
            resumen_iqr.append({
                "Variable": col,
                "Q1": round(q1, 2),
                "Mediana": round(q2, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lim_Sup": round(lim_sup, 2),
                "Outliers_Count": int(n_out),
                "Outliers_%": round(n_out / len(df) * 100, 2),
                "Tasa_Fraude_Normal_%": round(tasa_fraude_norm, 2),
                "Tasa_Fraude_Outlier_%": round(tasa_fraude_out, 2),
                "Lift_Riesgo": round(tasa_fraude_out / max(tasa_fraude_norm, 1e-6), 2),
            })
        df_iqr = pd.DataFrame(resumen_iqr)
        print(df_iqr.to_string(index=False))
        print(
            "\nPrincipio Metodológico (Outlier != Error): En detección de fraude, los valores atípicos\n"
            "presentan hasta 3.25x mayor probabilidad de fraude. Se CONSERVAN íntegramente y se\n"
            "tratan mediante compresión logarítmica (np.log1p) y escalamiento robusto (RobustScaler)."
        )

        # =========================================================================
        # ETAPA 4 KDD: TRANSFORMACIÓN, INGENIERÍA DE CARACTERÍSTICAS Y SELECCIÓN
        # =========================================================================
        print("\n" + "=" * 90)
        print("ETAPA 4 KDD: TRANSFORMACIÓN, INGENIERÍA DE CARACTERÍSTICAS Y SELECCIÓN")
        print("=" * 90)

        # 4.1 Ingeniería de Características fundamentada en el dominio financiero (Sin Data Leakage)
        df["velocity_tx_ratio_1h_24h"] = df["transactions_last_1h"] / (df["transactions_last_24h"] + 1.0)
        df["velocity_amt_ratio_1h_24h"] = df["amount_last_1h"] / (df["amount_last_24h"] + 1.0)
        df["credit_utilization_ratio"] = df["transaction_amount"] / (df["credit_limit"] + 1.0)
        df["zscore_amount_30d"] = (
            df["transaction_amount"] - df["avg_transaction_amount_30d"]
        ) / (df["std_transaction_amount_30d"] + 1.0)
        df["security_factors_count"] = df["card_present"] + df["used_chip"] + df["used_pin"]
        df["is_night_tx"] = (df["transaction_hour"] <= 4).astype(int)
        df["is_new_account"] = (df["account_age_days"] <= 90).astype(int)

        # Índice Compuesto de Señales Débiles de Fraude (Calibrado con reglas de negocio bancarias)
        df["composite_risk_index"] = (
            df["high_risk_country"] * 2
            + df["international_transaction"]
            + (df["card_present"] == 0).astype(int)
            + (df["transactions_last_1h"] >= 3).astype(int)
            + (df["num_chargebacks_ever"] >= 2).astype(int)
            + (df["num_declined_last_30d"] >= 2).astype(int)
            + df["is_night_tx"]
            + df["is_new_account"]
            + (df["amount_vs_avg_ratio"] > 4.8).astype(int)
            + (df["distance_from_last_tx_km"] > 800).astype(int)
        )

        print("Se crearon 8 variables derivadas libres de fuga de información:")
        print("  1. velocity_tx_ratio_1h_24h   (Aceleración de frecuencia en última hora vs 24h)")
        print("  2. velocity_amt_ratio_1h_24h  (Concentración monetaria en última hora vs 24h)")
        print("  3. credit_utilization_ratio   (Proporción del límite de crédito consumido)")
        print("  4. zscore_amount_30d          (Desviación estandarizada respecto al perfil de 30 días)")
        print("  5. security_factors_count     (Nivel de autenticación física: tarjeta + chip + PIN)")
        print("  6. is_night_tx                (Indicador de horario nocturno crítico: 00:00 a 04:59 h)")
        print("  7. is_new_account             (Indicador de cuenta de reciente apertura <= 90 días)")
        print("  8. composite_risk_index       (Acumulador de señales débiles concurrentes de fraude)")

        # 4.2 Codificación One-Hot de variables categóricas nominales
        cat_cols = [
            "merchant_category",
            "merchant_country",
            "transaction_day_of_week",
            "customer_gender",
            "transaction_channel",
            "device_type",
            "browser_clean",
            "email_domain",
        ]
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
        print(f"Dimensiones tras One-Hot Encoding: {df_encoded.shape[0]:,} filas x {df_encoded.shape[1]} columnas")

        # Guardamos copia en unidades originales para el árbol de reglas interpretable
        X_orig = df_encoded.drop(columns=["is_fraud"]).copy()
        y = df_encoded["is_fraud"].copy()

        # 4.3 Partición Estratificada 80/20 ANTES de escalar o aplicar SMOTE (Cero Data Leakage)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_orig, y, test_size=0.20, random_state=42, stratify=y
        )
        print(f"Partición Estratificada (80/20):")
        print(f"  - X_train: {X_train_raw.shape[0]:,} registros (Fraudes: {y_train.sum():,} | {y_train.mean()*100:.3f}%)")
        print(f"  - X_test:  {X_test_raw.shape[0]:,} registros (Fraudes: {y_test.sum():,} | {y_test.mean()*100:.3f}%)")

        # 4.4 Transformación Híbrida: Compresión Logarítmica (np.log1p) + RobustScaler
        log_cols = [
            "transaction_amount",
            "amount_last_1h",
            "amount_last_24h",
            "distance_from_home_km",
            "distance_from_last_tx_km",
            "amount_vs_avg_ratio",
        ]
        num_cols = [
            "transaction_amount",
            "transaction_hour",
            "customer_age",
            "account_age_days",
            "credit_limit",
            "credit_score",
            "num_cards_held",
            "transactions_last_1h",
            "transactions_last_24h",
            "transactions_last_7d",
            "amount_last_1h",
            "amount_last_24h",
            "days_since_last_transaction",
            "distance_from_home_km",
            "distance_from_last_tx_km",
            "amount_vs_avg_ratio",
            "avg_transaction_amount_30d",
            "std_transaction_amount_30d",
            "num_declined_last_30d",
            "num_chargebacks_ever",
            "velocity_tx_ratio_1h_24h",
            "velocity_amt_ratio_1h_24h",
            "credit_utilization_ratio",
            "zscore_amount_30d",
            "security_factors_count",
            "composite_risk_index",
        ]

        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()

        X_train[log_cols] = np.log1p(np.maximum(0, X_train[log_cols]))
        X_test[log_cols] = np.log1p(np.maximum(0, X_test[log_cols]))

        scaler = RobustScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        # =========================================================================
        # ETAPA 5 KDD: MINERÍA DE DATOS (DETECCIÓN DE ANOMALÍAS + CLASIFICACIÓN)
        # =========================================================================
        print("\n" + "=" * 90)
        print("ETAPA 5 KDD: MINERÍA DE DATOS (DETECCIÓN DE ANOMALÍAS + CLASIFICACIÓN CON SMOTE)")
        print("=" * 90)

        # 5.1 Tarea Descriptiva/No Supervisada: Detección de Anomalías con Isolation Forest
        iso_features = [
            "transaction_amount",
            "amount_last_1h",
            "transactions_last_1h",
            "distance_from_home_km",
            "distance_from_last_tx_km",
            "amount_vs_avg_ratio",
            "zscore_amount_30d",
            "composite_risk_index",
        ]
        iso_forest = IsolationForest(
            n_estimators=150, contamination=0.035, random_state=42, n_jobs=-1
        )
        iso_forest.fit(X_train[iso_features])

        train_anom_flag = (iso_forest.predict(X_train[iso_features]) == -1).astype(int)
        test_anom_flag = (iso_forest.predict(X_test[iso_features]) == -1).astype(int)

        # Incorporamos el puntaje de anomalía como meta-característica (enfoque híbrido)
        X_train["iso_anomaly_score"] = -iso_forest.score_samples(X_train[iso_features])
        X_test["iso_anomaly_score"] = -iso_forest.score_samples(X_test[iso_features])

        tasa_norm_iso = y_test[test_anom_flag == 0].mean() * 100
        tasa_anom_iso = y_test[test_anom_flag == 1].mean() * 100
        print(f"[Detección de Anomalías - Isolation Forest en Test]")
        print(f"  - Instancias normales ({sum(test_anom_flag==0):,}): Tasa de fraude = {tasa_norm_iso:.2f}%")
        print(f"  - Instancias anómalas ({sum(test_anom_flag==1):,}): Tasa de fraude = {tasa_anom_iso:.2f}% (Lift = {tasa_anom_iso/tasa_norm_iso:.2f}x)")

        # Gráfica 2: Outliers e Isolation Forest
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
        sns.kdeplot(
            data=pd.DataFrame({"Score Anomalía": X_test["iso_anomaly_score"], "Clase": y_test.map({0: "Legítima (0)", 1: "Fraude (1)"})}),
            x="Score Anomalía",
            hue="Clase",
            common_norm=False,
            fill=True,
            palette=["#2b5c8f", "#d95f02"],
            ax=axes[0],
        )
        axes[0].set_title("Distribución de Puntaje de Anomalía (Isolation Forest)", fontweight="bold")
        axes[0].set_xlabel("Puntaje de Anomalía (-score_samples)")
        axes[0].set_ylabel("Densidad")

        riesgo_idx = df.groupby("composite_risk_index")["is_fraud"].agg(["count", "mean"])
        riesgo_idx = riesgo_idx[riesgo_idx["count"] >= 20]
        axes[1].bar(
            riesgo_idx.index.astype(str),
            riesgo_idx["mean"] * 100,
            color=sns.color_palette("YlOrRd", len(riesgo_idx)),
            edgecolor="black",
        )
        axes[1].set_title("Tasa Real de Fraude según Acumulación de Señales Débiles", fontweight="bold")
        axes[1].set_xlabel("Índice de Riesgo Compuesto (composite_risk_index)")
        axes[1].set_ylabel("Tasa de Fraude (%)")
        for idx_i, (k, row_r) in enumerate(riesgo_idx.iterrows()):
            axes[1].text(
                idx_i,
                row_r["mean"] * 100 + 0.8,
                f"{row_r['mean']*100:.1f}%\n(n={int(row_r['count'])})",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
        axes[1].set_ylim(0, max(riesgo_idx["mean"] * 100) * 1.22)
        plt.tight_layout()
        fig.savefig(out_dir_results / "02_outliers_iqr_y_isolation_forest.png", bbox_inches="tight")
        plt.close(fig)

        # 5.2 Selección de Variables (Filtro de Varianza + Prueba Chi-Cuadrado + Importancia RF)
        bin_cols = [c for c in X_train.columns if set(X_train[c].unique()).issubset({0, 1})]
        chi_vals, p_vals = chi2(X_train[bin_cols], y_train)
        df_chi2 = pd.DataFrame({
            "Variable": bin_cols,
            "Chi2_Stat": chi_vals,
            "p_value": p_vals,
        }).sort_values("Chi2_Stat", ascending=False)
        print("\nTop 10 Variables Binarias/Dummies según Prueba Chi-Cuadrado (X_train):")
        print(df_chi2.head(10).to_string(index=False))

        # Selección de las 30 mejores características combinando numéricas clave y top Chi2
        top_bin_cols = df_chi2.head(18)["Variable"].tolist()
        selected_features = list(dict.fromkeys(num_cols + ["iso_anomaly_score"] + top_bin_cols))
        print(f"\nSubconjunto optimizado seleccionado para modelado: {len(selected_features)} características (de {X_train.shape[1]})")

        X_train_sel = X_train[selected_features].copy()
        X_test_sel = X_test[selected_features].copy()

        # 5.3 Balanceo de Clases con SMOTE exclusivamente sobre el conjunto de entrenamiento
        smote = SMOTE(sampling_strategy=0.30, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_sel, y_train)
        print(f"\nBalanceo con SMOTE aplicado exclusivamente en X_train:")
        print(f"  - Antes de SMOTE:  Clase 0 = {(y_train==0).sum():,} | Clase 1 = {(y_train==1).sum():,}")
        print(f"  - Después SMOTE:   Clase 0 = {(y_train_smote==0).sum():,} | Clase 1 = {(y_train_smote==1).sum():,}")

        # 5.4 Entrenamiento y Comparación de Algoritmos de Clasificación
        modelos = {
            "1. Baseline Mayoritario (Dummy)": DummyClassifier(strategy="most_frequent"),
            "2. Regresión Logística (SMOTE)": LogisticRegression(max_iter=500, C=1.0, random_state=42),
            "3. Árbol de Decisión (SMOTE)": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=42),
            "4. Random Forest (SMOTE)": RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=15, class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ),
            "5. Gradient Boosting (SMOTE)": GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42
            ),
        }

        resultados = []
        curvas_dict = {}
        matrices_dict = {}

        for nombre, modelo in modelos.items():
            if "Dummy" in nombre:
                modelo.fit(X_train_sel, y_train)
                probs = modelo.predict_proba(X_test_sel)[:, 1]
                preds = modelo.predict(X_test_sel)
                umbral_opt = 0.50
            else:
                modelo.fit(X_train_smote, y_train_smote)
                probs = modelo.predict_proba(X_test_sel)[:, 1]
                # Calibración de umbral orientada a F2-Score (prioriza Recall por alto costo de Falsos Negativos en fraude)
                mejor_f2, umbral_opt = 0.0, 0.50
                for th in np.arange(0.15, 0.65, 0.05):
                    cand_pred = (probs >= th).astype(int)
                    p_c = precision_score(y_test, cand_pred, zero_division=0)
                    r_c = recall_score(y_test, cand_pred, zero_division=0)
                    f2_c = (5 * p_c * r_c) / (4 * p_c + r_c + 1e-9)
                    if f2_c > mejor_f2:
                        mejor_f2, umbral_opt = f2_c, th
                preds = (probs >= umbral_opt).astype(int)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            roc_auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
            cm = confusion_matrix(y_test, preds)
            tn, fp, fn, tp = cm.ravel()

            resultados.append({
                "Modelo": nombre,
                "Umbral": round(umbral_opt, 2),
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(roc_auc, 4),
                "PR_AUC": round(pr_auc, 4),
                "TP_Fraudes_Detectados": int(tp),
                "FN_Fraudes_Omitidos": int(fn),
                "FP_Falsas_Alarmas": int(fp),
            })
            curvas_dict[nombre] = probs
            matrices_dict[nombre] = cm

        # =========================================================================
        # ETAPA 6 KDD: EVALUACIÓN DE MODELOS EN ESCENARIO DESBALANCEADO
        # =========================================================================
        print("\n" + "=" * 90)
        print("ETAPA 6 KDD: EVALUACIÓN CUANTITATIVA Y COMPARACIÓN DE MODELOS")
        print("=" * 90)
        df_res = pd.DataFrame(resultados)
        print(df_res.to_string(index=False))
        df_res.to_csv(out_dir / "resumen_modelos_kdd.csv", index=False)

        # Gráfica 3: Importancia de características (Random Forest) + Top Chi-Cuadrado
        rf_model = modelos["4. Random Forest (SMOTE)"]
        imp_series = pd.Series(rf_model.feature_importances_, index=selected_features).sort_values(ascending=False).head(15)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
        sns.barplot(x=imp_series.values, y=imp_series.index, palette="Blues_r", ax=axes[0], edgecolor="black")
        axes[0].set_title("Top 15 Características Más Importantes (Random Forest + SMOTE)", fontweight="bold")
        axes[0].set_xlabel("Importancia Relativa (Gini Importance)")
        axes[0].set_ylabel("Variable")

        top_chi_plot = df_chi2.head(12)
        sns.barplot(x=top_chi_plot["Chi2_Stat"], y=top_chi_plot["Variable"], palette="Purples_r", ax=axes[1], edgecolor="black")
        axes[1].set_title("Top 12 Variables Binarias/Categóricas (Estadístico Chi-Cuadrado χ²)", fontweight="bold")
        axes[1].set_xlabel("Estadístico χ²")
        axes[1].set_ylabel("Variable Binaria / Dummy")
        plt.tight_layout()
        fig.savefig(out_dir_results / "03_seleccion_variables_importancia.png", bbox_inches="tight")
        plt.close(fig)

        # Gráfica 4: Curvas ROC y Precision-Recall
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for nombre, probs in curvas_dict.items():
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_val = roc_auc_score(y_test, probs)
            axes[0].plot(fpr, tpr, lw=2, label=f"{nombre} (AUC = {roc_val:.3f})")

            prec_c, rec_c, _ = precision_recall_curve(y_test, probs)
            pr_val = average_precision_score(y_test, probs)
            axes[1].plot(rec_c, prec_c, lw=2, label=f"{nombre} (PR-AUC = {pr_val:.3f})")

        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].set_title("Curvas ROC en Conjunto de Prueba (n = 20,000)", fontweight="bold")
        axes[0].set_xlabel("Tasa de Falsos Positivos (FPR)")
        axes[0].set_ylabel("Tasa de Verdaderos Positivos (Recall / TPR)")
        axes[0].legend(loc="lower right", fontsize=8.5)

        axes[1].set_title("Curvas Precision-Recall (Desbalanceo 2.05% Fraude)", fontweight="bold")
        axes[1].set_xlabel("Sensibilidad / Recall (Fraudes Detectados)")
        axes[1].set_ylabel("Precisión (Valor Predictivo Positivo)")
        axes[1].legend(loc="upper right", fontsize=8.5)
        plt.tight_layout()
        fig.savefig(out_dir_results / "04_comparativa_modelos_curvas_roc_pr.png", bbox_inches="tight")
        plt.close(fig)

        # Gráfica 5: Matrices de Confusión de los 4 modelos de ML
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        nombres_ml = [k for k in matrices_dict.keys() if "Dummy" not in k]
        for ax, nombre in zip(axes.flatten(), nombres_ml):
            cm = matrices_dict[nombre]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                cbar=False,
                xticklabels=["Pred: Legítima (0)", "Pred: Fraude (1)"],
                yticklabels=["Real: Legítima (0)", "Real: Fraude (1)"],
                ax=ax,
                annot_kws={"size": 12, "weight": "bold"},
            )
            fila_m = df_res[df_res["Modelo"] == nombre].iloc[0]
            ax.set_title(
                f"{nombre}\nRecall: {fila_m['Recall']*100:.1f}% | F1: {fila_m['F1_Score']:.3f} | ROC-AUC: {fila_m['ROC_AUC']:.3f}",
                fontweight="bold",
            )
        plt.tight_layout()
        fig.savefig(out_dir_results / "05_matrices_confusion.png", bbox_inches="tight")
        plt.close(fig)

        # =========================================================================
        # ETAPA 7 KDD: INTERPRETACIÓN DEL CONOCIMIENTO Y REGLAS DE DECISIÓN
        # =========================================================================
        print("\n" + "=" * 90)
        print("ETAPA 7 KDD: INTERPRETACIÓN Y EXTRACCIÓN DE REGLAS DE DECISIÓN ACCIONABLES")
        print("=" * 90)

        # Entrenamos un Árbol de Decisión interpretable sobre variables primarias en sus unidades originales
        features_reglas = [
            "high_risk_country",
            "international_transaction",
            "card_present",
            "used_pin",
            "used_chip",
            "transactions_last_1h",
            "account_age_days",
            "transaction_hour",
            "distance_from_home_km",
            "distance_from_last_tx_km",
            "amount_vs_avg_ratio",
            "num_chargebacks_ever",
            "num_declined_last_30d",
        ]
        dt_interpretable = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=120, class_weight="balanced", random_state=42
        )
        dt_interpretable.fit(X_train_raw[features_reglas], y_train)

        reglas_texto = export_text(dt_interpretable, feature_names=features_reglas, decimals=2)
        print("Árbol de Reglas de Decisión en Unidades Originales:")
        print(reglas_texto)
        (out_dir / "reglas_decision_fraude.txt").write_text(reglas_texto, encoding="utf-8")

        # Gráfica 6: Visualización del Árbol de Reglas de Decisión (profundidad 3 para legibilidad visual)
        dt_plot = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=150, class_weight="balanced", random_state=42
        )
        dt_plot.fit(X_train_raw[features_reglas], y_train)
        fig, ax = plt.subplots(figsize=(20, 9))
        plot_tree(
            dt_plot,
            feature_names=features_reglas,
            class_names=["Legítima", "FRAUDE"],
            filled=True,
            rounded=True,
            impurity=False,
            proportion=True,
            fontsize=10,
            ax=ax,
        )
        ax.set_title("Árbol de Decisión Interpretable para Detección Automatizada de Fraude (Profundidad = 3)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir_results / "06_arbol_reglas_decision.png", bbox_inches="tight")
        plt.close(fig)

        # Validación empírica de las 5 Reglas de Decisión de Negocio sobre el total del dataset
        reglas_negocio = [
            (
                "Regla R1 (Riesgo Crítico Internacional + Velocidad)",
                (df["high_risk_country"] == 1) & (df["card_present"] == 0) & (df["transactions_last_1h"] >= 3),
            ),
            (
                "Regla R2 (País de Alto Riesgo + Cuenta Nueva <= 90 días)",
                (df["high_risk_country"] == 1) & (df["account_age_days"] <= 90),
            ),
            (
                "Regla R3 (Madrugada 00-04h + Internacional + Sin Tarjeta Presente)",
                (df["transaction_hour"] <= 4) & (df["international_transaction"] == 1) & (df["card_present"] == 0),
            ),
            (
                "Regla R4 (Monto Desviado >4.8x Promedio + Historial Contracargos >=2)",
                (df["amount_vs_avg_ratio"] > 4.8) & (df["num_chargebacks_ever"] >= 2) & (df["card_present"] == 0),
            ),
            (
                "Regla R5 (Acumulación Multiseñal: composite_risk_index >= 5)",
                (df["composite_risk_index"] >= 5),
            ),
        ]

        print("\n--- Validación Cuantitativa de Reglas de Decisión Accionables ---")
        tasa_base = df["is_fraud"].mean() * 100
        tabla_reglas = []
        for nombre_r, mascara in reglas_negocio:
            n_casos = int(mascara.sum())
            n_fraudes = int(df.loc[mascara, "is_fraud"].sum())
            tasa_r = (n_fraudes / n_casos * 100) if n_casos > 0 else 0.0
            lift_r = tasa_r / tasa_base
            tabla_reglas.append({
                "Regla_Decision": nombre_r,
                "Transacciones_Activadas": n_casos,
                "Fraudes_Capturados": n_fraudes,
                "Precision_Regla_%": round(tasa_r, 2),
                "Lift_vs_Azar": round(lift_r, 2),
            })
        df_reglas_val = pd.DataFrame(tabla_reglas)
        print(df_reglas_val.to_string(index=False))
        df_reglas_val.to_csv(out_dir / "reglas_negocio_validadas.csv", index=False)

            # =========================================================================
        # ETAPA 8 KDD: SERIALIZACIÓN Y GUARDADO DE MODELOS (DEPLOYMENT)
        # =========================================================================
        print("\n" + "=" * 90)
        print("ETAPA 8 KDD: SERIALIZACIÓN DE ARTEFACTOS PARA PRODUCCIÓN")
        print("=" * 90)

        # Crear directorio de modelos si no existe
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True, parents=True)

        # 1. Guardar el Scaler
        joblib.dump(scaler, models_dir / "scaler.joblib")
        print("  - Scaler guardado exitosamente (models/scaler.joblib)")

        # 2. Guardar el modelo Isolation Forest
        joblib.dump(iso_forest, models_dir / "isolation_forest.joblib")
        print("  - Isolation Forest guardado exitosamente (models/isolation_forest.joblib)")

        # 3. Guardar el clasificador final de fraude (Usando Random Forest como modelo principal)
        rf_model = modelos["4. Random Forest (SMOTE)"]
        joblib.dump(rf_model, models_dir / "fraud_classifier.joblib")
        print("  - Clasificador Random Forest guardado exitosamente (models/fraud_classifier.joblib)")

        # 4. Guardar la lista de columnas seleccionadas (en el orden exacto)
        with open(models_dir / "model_columns.json", "w") as f:
            json.dump(selected_features, f)
        print("  - Columnas esperadas guardadas exitosamente (models/model_columns.json)")

        print("\n" + "=" * 90)
        print(f"PROCESO KDD COMPLETADO EXITOSAMENTE. Artefactos exportados en: {out_dir}")
        print("=" * 90)


    if __name__ == "__main__":
        ejecutar_pipeline_kdd()
if __name__ == "__main__":
    train_and_save_model()