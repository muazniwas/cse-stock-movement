export type SymbolListResponse = {
  count: number;
  symbols: string[];
};

export type FeatureMeta = {
  name: string;
  description?: string;
  category?: string;
};

export type ModelInfoResponse = {
  model_type: string;
  model_file: string;
  threshold: number;
  min_history_rows: number;
  feature_count: number;
  features: FeatureMeta[];
  symbols_count: number;
  symbols: string[];
};

export type ConfusionMatrix = {
  labels: string[];
  matrix: number[][];
};

export type ModelMetricsResponse = {
  test_samples: number;
  roc_auc: number;
  accuracy: number;
  confusion_matrix: ConfusionMatrix;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  classification_report: Record<string, any>; // sklearn dict;
  class_distribution: Record<string, number>;
};

export type OhlcvRow = {
  date?: string;  // "YYYY-MM-DD"
  low: number;
  high: number;
  close: number;
  volume: number;
};

export type HistoryResponse = {
  symbol: string;      // normalized like "JKH.N0000"
  stock_id: number;
  period: number;
  count: number;
  history: OhlcvRow[]; // oldest -> newest
};

export type PredictRequest = {
  symbol: string;      // eg: "JKH"
  history: OhlcvRow[];
};

export type PredictResponse = {
  symbol: string;
  prob_up: number;
  prediction: 0 | 1;
  threshold: number;
};

export type ExplainContribution = {
  feature: string;
  value: number;
  shap_value: number;
};

export type ExplainResponse = {
  symbol: string;
  prob_up: number;
  prediction: 0 | 1;
  threshold: number;
  base_value: number;
  top_contributions: ExplainContribution[];
  all_contributions?: ExplainContribution[];
};
