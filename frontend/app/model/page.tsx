"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ModelInfoResponse, ModelMetricsResponse } from "@/lib/api/types";

export default function ModelPage() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        setError(null);

        const [i, m] = await Promise.all([api.modelInfo(), api.modelMetrics()]);
        if (cancelled) return;

        setInfo(i);
        setMetrics(m);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } catch (e: any) {
        if (cancelled) return;
        setError(e.message || "Failed to load model details");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Model</h1>
        {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
        <a className="text-blue-600 underline" href="/">
          ← Back to prediction
        </a>
      </div>

      {loading && <p>Loading…</p>}

      {error && (
        <div className="p-3 bg-red-100 text-red-700 border border-red-300 rounded">
          {error}
        </div>
      )}

      {/* --- Model info --- */}
      {info && (
        <section className="border rounded p-4 bg-gray-50 space-y-2">
          <h2 className="font-semibold text-lg text-black">Model Info</h2>
          <div className="text-sm space-y-1">
            <p className="text-black">
              <span className="font-medium">Type:</span> {info.model_type}
            </p>
            <p className="text-black">
              <span className="font-medium">Model file:</span> {info.model_file}
            </p>
            <p className="text-black">
              <span className="font-medium">Threshold:</span> {info.threshold}
            </p>
            <p className="text-black">
              <span className="font-medium">Minimum history rows:</span>{" "}
              {info.min_history_rows}
            </p>
            <p className="text-black">
              <span className="font-medium">Features:</span> {info.feature_count}
            </p>
            <p className="text-black">
              <span className="font-medium">Symbols:</span> {info.symbols_count}
            </p>
          </div>
        </section>
      )}

      {/* --- Metrics --- */}
      {metrics && (
        <section className="border rounded p-4 bg-gray-50 space-y-4">
          <h2 className="font-semibold text-lg text-black">Evaluation Metrics (Test Set)</h2>

          <div className="text-sm space-y-1">
            <p className="text-black">
              <span className="font-medium">Test samples:</span>{" "}
              {metrics.test_samples}
            </p>
            <p className="text-black">
              <span className="font-medium">ROC-AUC:</span> {metrics.roc_auc}
            </p>
            <p className="text-black">
              <span className="font-medium">Accuracy:</span> {metrics.accuracy}
            </p>
            <p className="text-black">
              <span className="font-medium">Class distribution:</span>{" "}
              {JSON.stringify(metrics.class_distribution)}
            </p>
          </div>

          {/* Confusion matrix */}
          <div>
            <h3 className="font-semibold text-black mb-2">Confusion Matrix</h3>
            <div className="overflow-auto border rounded">
              <table className="w-full text-sm border-collapse text-center">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="text-black border px-2 py-1"></th>
                    {metrics.confusion_matrix.labels.map((lbl) => (
                      <th key={lbl} className="text-black border px-2 py-1">
                        Pred {lbl}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.confusion_matrix.matrix.map((row, i) => (
                    <tr key={i}>
                      <td className="text-black border px-2 py-1 font-medium bg-gray-100">
                        Actual {metrics.confusion_matrix.labels[i]}
                      </td>
                      {row.map((val, j) => (
                        <td key={j} className="text-black border px-2 py-1">
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {/* --- Feature list with descriptions --- */}
      {info && (
        <section className="space-y-2">
          <h2 className="font-semibold text-lg">Feature Definitions</h2>
          <div className="overflow-auto border rounded">
            <table className="w-full text-sm border-collapse">
              <thead className="bg-gray-100">
                <tr>
                  <th className="border px-2 py-1 text-left text-black">Feature</th>
                  <th className="border px-2 py-1 text-left text-black">Category</th>
                  <th className="border px-2 py-1 text-left text-black">Description</th>
                </tr>
              </thead>
              <tbody>
                {info.features.map((f) => (
                  <tr key={f.name}>
                    <td className="border px-2 py-1 font-mono">{f.name}</td>
                    <td className="border px-2 py-1">{f.category || "-"}</td>
                    <td className="border px-2 py-1">{f.description || "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}
