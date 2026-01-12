import { apiFetch } from "./client";
import type {
  SymbolListResponse,
  ModelInfoResponse,
  ModelMetricsResponse,
  HistoryResponse,
  PredictRequest,
  PredictResponse,
  ExplainResponse,
} from "./types";

export const api = {
  health: () => apiFetch<{ status: string }>("/health"),

  symbols: () => apiFetch<SymbolListResponse>("/symbols"),

  modelInfo: () => apiFetch<ModelInfoResponse>("/model/info"),

  modelMetrics: () => apiFetch<ModelMetricsResponse>("/model/metrics"),

  history: (symbol: string, n: number = 15, period: number = 5) =>
    apiFetch<HistoryResponse>(
      `/history/${encodeURIComponent(symbol)}?n=${encodeURIComponent(n)}&period=${encodeURIComponent(period)}`
    ),

  predict: (payload: PredictRequest) =>
    apiFetch<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  explain: (payload: PredictRequest) =>
    apiFetch<ExplainResponse>("/explain", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
