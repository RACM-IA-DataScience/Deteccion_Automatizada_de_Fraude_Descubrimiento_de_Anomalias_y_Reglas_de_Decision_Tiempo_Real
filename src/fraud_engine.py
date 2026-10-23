"""
Motor Central de Inferencia y Alertas de Fraude (FraudGuard Engine)
Módulo reusable para evaluación de transacciones en tiempo real,
descubrimiento de anomalías, reglas de negocio determinísticas y despacho de alertas.
"""

from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import joblib


class FraudInferenceEngine:
    """
    Motor integral de scoring de fraude que combina:
    1. Ingeniería de características financieras en vuelo.
    2. Evaluación de reglas de negocio duras (R1 a R5).
    3. Detección no supervisada de anomalías (Isolation Forest).
    4. Clasificación supervisada de probabilidad de fraude (Random Forest).
    5. Motor de decisión y generación de alertas operativas.
    """

    def __init__(self, models_dir: Path = None):
        if models_dir is None:
            # Buscar en el directorio models relativo a este archivo o la raíz
            current_dir = Path(__file__).resolve().parent
            candidates = [
                current_dir.parent / "models",
                current_dir / "models",
                Path("models"),
                Path("/opt/airflow/models")
            ]
            for cand in candidates:
                if (cand / "fraud_classifier.joblib").exists():
                    models_dir = cand
                    break

        if models_dir is None or not (models_dir / "fraud_classifier.joblib").exists():
            raise FileNotFoundError(f"No se encontró el directorio de modelos en: {models_dir}")

        self.models_dir = models_dir
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        self.iso_forest = joblib.load(models_dir / "isolation_forest.joblib")
        self.classifier = joblib.load(models_dir / "fraud_classifier.joblib")
        with open(models_dir / "model_columns.json", "r", encoding="utf-8") as f:
            self.model_columns = json.load(f)

        self.log_cols = [
            "transaction_amount",
            "amount_last_1h",
            "amount_last_24h",
            "distance_from_home_km",
            "distance_from_last_tx_km",
            "amount_vs_avg_ratio"
        ]

        self.num_cols = list(getattr(self.scaler, "feature_names_in_", []))
        self.iso_features = list(getattr(self.iso_forest, "feature_names_in_", []))

    def preprocess_and_feature_engineering(self, raw_tx: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calcula las variables derivadas, ratios de velocidad y factores de autenticación.
        """
        df = pd.DataFrame([raw_tx]).copy()

        # Normalizar identificadores
        if "transaction_id" not in df.columns or not df["transaction_id"].iloc[0]:
            df["transaction_id"] = f"TXN-{uuid.uuid4().hex[:8].upper()}"

        if "transaction_datetime" not in df.columns:
            df["transaction_datetime"] = datetime.now(timezone.utc)

        # Mapeo y saneamiento de browser
        if "browser" in df.columns:
            df = df.rename(columns={"browser": "browser_clean"})
        elif "browser_clean" not in df.columns:
            df["browser_clean"] = "No_Aplica_Fisico"
        df["browser_clean"] = df["browser_clean"].fillna("No_Aplica_Fisico")

        # Variables derivadas de dominio financiero
        df["velocity_tx_ratio_1h_24h"] = df["transactions_last_1h"] / (df["transactions_last_24h"] + 1.0)
        df["velocity_amt_ratio_1h_24h"] = df["amount_last_1h"] / (df["amount_last_24h"] + 1.0)
        df["credit_utilization_ratio"] = df["transaction_amount"] / (df["credit_limit"] + 1.0)
        df["zscore_amount_30d"] = (
            df["transaction_amount"] - df["avg_transaction_amount_30d"]
        ) / (df["std_transaction_amount_30d"] + 1.0)
        df["security_factors_count"] = df["card_present"] + df["used_chip"] + df["used_pin"]
        df["is_night_tx"] = (df["transaction_hour"] <= 4).astype(int)
        df["is_new_account"] = (df["account_age_days"] <= 90).astype(int)

        # Índice Compuesto de Señales Débiles (composite_risk_index)
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

        engineered_meta = {
            "velocity_tx_ratio_1h_24h": round(float(df["velocity_tx_ratio_1h_24h"].iloc[0]), 3),
            "velocity_amt_ratio_1h_24h": round(float(df["velocity_amt_ratio_1h_24h"].iloc[0]), 3),
            "credit_utilization_ratio": round(float(df["credit_utilization_ratio"].iloc[0]), 3),
            "zscore_amount_30d": round(float(df["zscore_amount_30d"].iloc[0]), 3),
            "security_factors_count": int(df["security_factors_count"].iloc[0]),
            "composite_risk_index": int(df["composite_risk_index"].iloc[0]),
            "is_night_tx": bool(df["is_night_tx"].iloc[0]),
            "is_new_account": bool(df["is_new_account"].iloc[0]),
        }

        return df, engineered_meta

    def evaluate_business_rules(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Evalúa las 5 reglas de negocio validadas en el proyecto KDD.
        """
        row = df.iloc[0]
        rules_status = []

        # R1: Alto riesgo internacional + sin tarjeta + alta velocidad
        r1 = bool(
            (row.get("high_risk_country", 0) == 1)
            and (row.get("card_present", 1) == 0)
            and (row.get("transactions_last_1h", 0) >= 3)
        )
        rules_status.append({
            "code": "R1",
            "name": "Riesgo Crítico Internacional y Velocidad",
            "description": "País de alto riesgo + Sin tarjeta presente + >=3 txs en última hora",
            "triggered": r1,
            "severity": "CRITICAL"
        })

        # R2: País de alto riesgo + cuenta recién abierta
        r2 = bool(
            (row.get("high_risk_country", 0) == 1)
            and (row.get("account_age_days", 999) <= 90)
        )
        rules_status.append({
            "code": "R2",
            "name": "País de Alto Riesgo en Cuenta Nueva",
            "description": "País de alto riesgo + Antigüedad de cuenta <= 90 días",
            "triggered": r2,
            "severity": "HIGH"
        })

        # R3: Madrugada + Internacional + Sin tarjeta presente
        r3 = bool(
            (row.get("transaction_hour", 12) <= 4)
            and (row.get("international_transaction", 0) == 1)
            and (row.get("card_present", 1) == 0)
        )
        rules_status.append({
            "code": "R3",
            "name": "Madrugada Internacional No Presencial",
            "description": "Horario 00-04h + Tx internacional + Sin tarjeta presente",
            "triggered": r3,
            "severity": "HIGH"
        })

        # R4: Monto anómalo vs promedio histórico + historial de contracargos
        r4 = bool(
            (row.get("amount_vs_avg_ratio", 1.0) > 4.8)
            and (row.get("num_chargebacks_ever", 0) >= 2)
            and (row.get("card_present", 1) == 0)
        )
        rules_status.append({
            "code": "R4",
            "name": "Monto Desviado con Contracargos Previos",
            "description": "Monto >4.8x promedio 30d + >=2 contracargos + Sin tarjeta presente",
            "triggered": r4,
            "severity": "HIGH"
        })

        # R5: Desbordamiento del índice de riesgo compuesto
        r5 = bool(row.get("composite_risk_index", 0) >= 5)
        rules_status.append({
            "code": "R5",
            "name": "Acumulación Multiseñal de Riesgo (Índice >= 5)",
            "description": "Índice de riesgo compuesto >= 5 señales concurrentes",
            "triggered": r5,
            "severity": "CRITICAL"
        })

        return rules_status

    def predict(self, raw_tx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de inferencia sobre una transacción individual.
        """
        df_proc, meta = self.preprocess_and_feature_engineering(raw_tx)
        rules = self.evaluate_business_rules(df_proc)
        critical_rule_triggered = any(r["triggered"] for r in rules if r["code"] in ["R1", "R5"])
        any_rule_triggered = any(r["triggered"] for r in rules)

        # 1. Preparar DataFrame de variables numéricas y calculadas
        df_transformed = df_proc.copy()

        # Asegurar que todas las columnas numéricas existan
        for col in self.num_cols:
            if col not in df_transformed.columns:
                df_transformed[col] = 0.0

        # Transformación logarítmica previa al escalamiento (igual que en entrenamiento)
        for c in self.log_cols:
            if c in df_transformed.columns:
                df_transformed[c] = np.log1p(np.maximum(0, df_transformed[c].astype(float)))

        # Escalamiento robusto
        scaled_values = self.scaler.transform(df_transformed[self.num_cols])
        df_scaled_nums = pd.DataFrame(scaled_values, columns=self.num_cols, index=df_transformed.index)

        # 2. Inferencia de Isolation Forest (Anomalía No Supervisada sobre features escaladas)
        iso_score = float(-self.iso_forest.score_samples(df_scaled_nums[self.iso_features])[0])
        is_anomaly = bool(self.iso_forest.predict(df_scaled_nums[self.iso_features])[0] == -1)

        # 3. Construir vector de características exacto para el clasificador supervisado
        # Mapeo determinístico de variables categóricas One-Hot (evitando el drop_first en inferencia unitaria)
        feature_dict = {}
        for col in self.model_columns:
            if col in self.num_cols:
                feature_dict[col] = float(df_scaled_nums[col].iloc[0])
            elif col == "iso_anomaly_score":
                feature_dict[col] = iso_score
            elif col in df_proc.columns:
                # Variable binaria directa (ej. card_present, used_chip, high_risk_country)
                feature_dict[col] = int(df_proc[col].iloc[0])
            else:
                # Dummy generada de categórica: formato {columna_original}_{valor}
                # Buscar si alguna columna categórica coincide
                matched = False
                for cat_field in ["merchant_country", "merchant_category", "transaction_channel", "device_type", "browser_clean", "transaction_day_of_week", "customer_gender", "email_domain"]:
                    if col.startswith(f"{cat_field}_"):
                        val_in_tx = str(df_proc[cat_field].iloc[0]) if cat_field in df_proc.columns else ""
                        expected_val = col[len(f"{cat_field}_"):]
                        if val_in_tx == expected_val:
                            feature_dict[col] = 1
                            matched = True
                            break
                if not matched:
                    feature_dict[col] = 0

        features_for_model = pd.DataFrame([feature_dict])[self.model_columns]
        fraud_prob = float(self.classifier.predict_proba(features_for_model)[:, 1][0])
        model_pred = int(fraud_prob >= 0.50)

        # 5. Motor de Decisión Tripartito
        if critical_rule_triggered or fraud_prob >= 0.70:
            recommended_action = "DECLINE"
            status_desc = "Transacción Bloqueada Inmediatamente"
            severity = "CRITICAL"
        elif fraud_prob >= 0.35 or is_anomaly or any_rule_triggered:
            recommended_action = "MANUAL_REVIEW"
            status_desc = "Transacción Marcada para Auditoría Manual / 2FA"
            severity = "WARNING"
        else:
            recommended_action = "APPROVE"
            status_desc = "Transacción Aprobada Exitosamente"
            severity = "LOW"

        # 6. Generación del Payload de Alerta Simulada
        tx_id = str(df_proc["transaction_id"].iloc[0])
        alert_payload = self._build_alert_payload(
            tx_id=tx_id,
            raw_tx=raw_tx,
            fraud_prob=fraud_prob,
            iso_score=iso_score,
            is_anomaly=is_anomaly,
            rules=rules,
            recommended_action=recommended_action,
            severity=severity
        )

        return {
            "transaction_id": tx_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommended_action": recommended_action,
            "status_description": status_desc,
            "severity": severity,
            "fraud_probability": round(fraud_prob, 4),
            "model_prediction": model_pred,
            "anomaly_score": round(iso_score, 4),
            "is_anomaly": is_anomaly,
            "rules_evaluated": rules,
            "engineered_features": meta,
            "alert": alert_payload
        }

    def _build_alert_payload(
        self,
        tx_id: str,
        raw_tx: Dict[str, Any],
        fraud_prob: float,
        iso_score: float,
        is_anomaly: bool,
        rules: List[Dict[str, Any]],
        recommended_action: str,
        severity: str
    ) -> Dict[str, Any]:
        """Construye un payload estructurado simulando una notificación a sistemas de alerta."""
        triggered_rules = [r["code"] + ": " + r["name"] for r in rules if r["triggered"]]
        amount = raw_tx.get("transaction_amount", 0.0)
        merchant = raw_tx.get("merchant_category", "Comercio no especificado")
        country = raw_tx.get("merchant_country", "Desconocido")

        if recommended_action == "DECLINE":
            sms_text = (
                f"🚨 BANCO ALERTA: Cargo sospechoso RECHAZADO por ${amount:,.2f} en {merchant} ({country}). "
                f"Tu tarjeta ha sido bloqueada temporalmente por seguridad. Si fuiste tú, responde SI al 8800."
            )
        elif recommended_action == "MANUAL_REVIEW":
            sms_text = (
                f"⚠️ BANCO ALERTA: Detectamos una transacción inusual por ${amount:,.2f} en {merchant}. "
                f"Por favor ingresa a tu App Bancaria para aprobar este movimiento con tu Token Digital."
            )
        else:
            sms_text = f"✅ BANCO NOTIFICACIÓN: Cargo aprobado por ${amount:,.2f} en {merchant}."

        return {
            "alert_id": f"ALT-{uuid.uuid4().hex[:8].upper()}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "transaction_id": tx_id,
            "severity": severity,
            "action_taken": recommended_action,
            "summary": f"Evaluación de riesgo: {recommended_action} (Probabilidad: {fraud_prob*100:.1f}%, Anomalía: {iso_score:.2f})",
            "reasons": triggered_rules if triggered_rules else ["Probabilidad dentro del rango de tolerancia normal"],
            "simulated_sms": sms_text,
            "simulated_webhook": {
                "source": "FraudGuard-KDD-Engine",
                "event": "transaction_risk_evaluated",
                "severity": severity,
                "transaction": {
                    "id": tx_id,
                    "amount": amount,
                    "merchant_category": merchant,
                    "country": country,
                },
                "metrics": {
                    "fraud_probability": round(fraud_prob, 4),
                    "anomaly_score": round(iso_score, 4),
                    "is_isolation_anomaly": is_anomaly
                },
                "rules_violated": [r["code"] for r in rules if r["triggered"]]
            }
        }
