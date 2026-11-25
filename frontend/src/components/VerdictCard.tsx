/* ── Verdict card — prediction result ── */
import { AlertTriangle, CheckCircle2, ShieldAlert, Clock } from "lucide-react";
import type { PredictionResponse } from "../api/types";

interface Props {
  result: PredictionResponse;
}

export default function VerdictCard({ result }: Props) {
  const riskColor = result.is_abnormal
    ? "from-red-500/20 to-red-900/10 border-red-700/40"
    : "from-emerald-500/20 to-emerald-900/10 border-emerald-700/40";

  const riskText = result.is_abnormal ? "Abnormal Finding" : "Normal Finding";
  const RiskIcon = result.is_abnormal ? ShieldAlert : CheckCircle2;

  const confidenceColor =
    result.confidence >= 0.85
      ? "text-emerald-400"
      : result.confidence >= 0.6
      ? "text-yellow-400"
      : "text-red-400";

  return (
    <div
      className={`glass-card p-5 bg-gradient-to-br ${riskColor} animate-fade-in`}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider font-medium">
            {result.display_name}
          </p>
          <h3 className="text-2xl font-bold text-gray-100 mt-1">{result.prediction}</h3>
        </div>
        <div
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${
            result.is_abnormal
              ? "border-red-600/50 bg-red-900/30 text-red-400"
              : "border-emerald-600/50 bg-emerald-900/30 text-emerald-400"
          }`}
        >
          <RiskIcon className="w-3.5 h-3.5" />
          {riskText}
        </div>
      </div>

      {/* Confidence */}
      <div className="flex items-center gap-4 mb-4">
        <div>
          <p className="text-xs text-gray-500">Confidence</p>
          <p className={`text-xl font-bold ${confidenceColor}`}>
            {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>
        <div className="flex-1 h-2 rounded-full bg-gray-800 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              result.is_abnormal ? "bg-red-500" : "bg-emerald-500"
            }`}
            style={{ width: `${result.confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Meta */}
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {result.inference_ms.toFixed(0)} ms
        </span>
        <span>Device: {result.device.toUpperCase()}</span>
        {result.threshold_used !== null && (
          <span>Threshold: {(result.threshold_used * 100).toFixed(0)}%</span>
        )}
      </div>

      {/* Warnings */}
      {result.warnings.length > 0 && (
        <div className="mt-3 space-y-1">
          {result.warnings.map((w, i) => (
            <div
              key={i}
              className="flex items-start gap-2 text-xs text-yellow-400/80 bg-yellow-900/10 border border-yellow-800/30 rounded-lg px-3 py-2"
            >
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              {w}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
