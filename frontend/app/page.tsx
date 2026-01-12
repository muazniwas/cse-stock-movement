"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type {
  OhlcvRow,
  PredictResponse,
  ExplainResponse,
} from "@/lib/api/types";

export default function HomePage() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");

  const [history, setHistory] = useState<OhlcvRow[] | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);

  const [error, setError] = useState<string | null>(null);

  // Load symbols on first load
  useEffect(() => {
    api.symbols()
      .then((res) => {
        setSymbols(res.symbols);
        if (res.symbols.length > 0) {
          setSelectedSymbol(res.symbols[0]);
        }
      })
      .catch((e) => setError(e.message));
  }, []);

  // Load history when symbol changes
  useEffect(() => {
    if (!selectedSymbol) return;

    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoadingHistory(true);
    setError(null);
    setPrediction(null);
    setExplanation(null);

    api.history(selectedSymbol, 15)
      .then((res) => {
        setHistory(res.history);
      })
      .catch((e) => {
        setError(e.message);
        setHistory(null);
      })
      .finally(() => setLoadingHistory(false));
  }, [selectedSymbol]);

  async function handlePredict() {
    if (!history || !selectedSymbol) return;

    try {
      setError(null);

      const payload = {
        symbol: selectedSymbol,
        history,
      };

      const pred = await api.predict(payload);
      const exp = await api.explain(payload);

      setPrediction(pred);
      setExplanation(exp);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (e: any) {
      setError(e.message || "Prediction failed");
    }
  }

  return (
    <main className="p-6 max-w-6xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">
        CSE Stock Movement Prediction (LightGBM)
      </h1>

      {/* --- Symbol selector --- */}
      <div className="flex items-center gap-4">
        <label className="font-medium">Symbol:</label>
        <select
          value={selectedSymbol}
          onChange={(e) => setSelectedSymbol(e.target.value)}
          className="border px-3 py-1 rounded"
        >
          {symbols.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        {loadingHistory && <span className="text-sm">Loading history…</span>}
      </div>

      {/* --- Error --- */}
      {error && (
        <div className="p-3 bg-red-100 text-red-700 border border-red-300 rounded">
          {error}
        </div>
      )}

      {/* --- History table --- */}
      {history && (
        <div>
          <h2 className="font-semibold mb-2">Recent history (used as input)</h2>
          <div className="overflow-auto border rounded">
            <table className="w-full text-sm border-collapse">
              <thead className="bg-gray-100">
                <tr>
                  <th className="font-semibold mb-2 text-black border px-2 py-1">Date</th>
                  <th className="font-semibold mb-2 text-black border px-2 py-1">Low</th>
                  <th className="font-semibold mb-2 text-black border px-2 py-1">High</th>
                  <th className="font-semibold mb-2 text-black border px-2 py-1">Close</th>
                  <th className="font-semibold mb-2 text-black border px-2 py-1">Volume</th>
                </tr>
              </thead>
              <tbody>
                {history.map((r, i) => (
                  <tr key={i} className="text-center">
                    <td className="border px-2 py-1">{r.date}</td>
                    <td className="border px-2 py-1">{r.low}</td>
                    <td className="border px-2 py-1">{r.high}</td>
                    <td className="border px-2 py-1">{r.close}</td>
                    <td className="border px-2 py-1">{r.volume}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* --- Predict button --- */}
      <button
        onClick={handlePredict}
        disabled={!history}
        className="px-5 py-2 bg-black font-semibold mb-2 text-white rounded hover:bg-gray-800 disabled:opacity-50"
      >
        Predict next-day movement
      </button>

      {/* --- Prediction output --- */}
      {prediction && (
        <div className="p-4 border rounded bg-gray-50">
          <h2 className="font-semibold text-black mb-2">Prediction</h2>
          <p className="text-black">
            Direction:{" "}
            <span className="font-bold">
              {prediction.prediction === 1 ? "UP 📈" : "DOWN 📉"}
            </span>
          </p>
          <p className="text-black">Probability of UP: {(prediction.prob_up * 100).toFixed(2)}%</p>
        </div>
      )}

      {/* --- Explainability output --- */}
      {explanation && (
        <div className="p-4 border rounded bg-gray-50">
          <h2 className="font-semibold mb-2 text-black">Top feature contributions (SHAP)</h2>
          <table className="w-full text-sm border-collapse">
            <thead className="bg-gray-100">
              <tr>
                <th className="text-black border px-2 py-1">Feature</th>
                <th className="text-black border px-2 py-1">Value</th>
                <th className="text-black border px-2 py-1">SHAP</th>
              </tr>
            </thead>
            <tbody>
              {explanation.top_contributions.map((c, i) => (
                <tr key={i} className="text-center">
                  <td className="text-black border px-2 py-1">{c.feature}</td>
                  <td className="text-black border px-2 py-1">{c.value}</td>
                  <td
                    className={`border px-2 py-1 ${
                      c.shap_value >= 0 ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {c.shap_value.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
