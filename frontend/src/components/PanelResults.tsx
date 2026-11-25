/* ── Panel results — shows results from multiple models ── */
import { AlertTriangle, CheckCircle2, Clock, ShieldAlert, XCircle } from "lucide-react";
import type { PanelResult } from "../api/types";

interface Props {
  results: PanelResult[];
}

export default function PanelResults({ results }: Props) {
  const successful = results.filter((r) => !r.error);
  const failed = results.filter((r) => r.error);

  return (
    <div className="space-y-3 animate-fade-in">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
        Panel Results — {successful.length} of {results.length} completed
      </h3>

      {successful.map((r) => {
        const isAbnormal = r.is_abnormal ?? false;
        return (
          <div
            key={r.task_id}
            className={`glass-card p-4 border-l-4 ${
              isAbnormal ? "border-l-red-500" : "border-l-emerald-500"
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <p className="text-xs text-gray-500 uppercase tracking-wider font-medium">
                    {r.display_name}
                  </p>
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                      isAbnormal
                        ? "bg-red-900/30 text-red-400 border border-red-800/40"
                        : "bg-emerald-900/30 text-emerald-400 border border-emerald-800/40"
                    }`}
                  >
                    {isAbnormal ? (
                      <ShieldAlert className="w-2.5 h-2.5" />
                    ) : (
                      <CheckCircle2 className="w-2.5 h-2.5" />
                    )}
                    {isAbnormal ? "Abnormal" : "Normal"}
                  </span>
                </div>
                <h4 className="text-lg font-bold text-gray-100">{r.prediction}</h4>
              </div>
              <div className="text-right shrink-0">
                <p className="text-xs text-gray-500">Confidence</p>
                <p
                  className={`text-xl font-bold tabular-nums ${
                    (r.confidence ?? 0) >= 0.85
                      ? "text-emerald-400"
                      : (r.confidence ?? 0) >= 0.6
                      ? "text-yellow-400"
                      : "text-red-400"
                  }`}
                >
                  {((r.confidence ?? 0) * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Probability mini-bars */}
            {r.probabilities && (
              <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5">
                {Object.entries(r.probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, prob]) => (
                    <div key={cls} className="flex items-center gap-2">
                      <span className="text-[11px] text-gray-500 w-32 truncate">{cls}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-gray-800 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            cls === r.prediction
                              ? "bg-clinical-500"
                              : "bg-gray-700"
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                      <span className="text-[10px] text-gray-600 tabular-nums w-10 text-right">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
              </div>
            )}

            {/* Meta row */}
            <div className="flex items-center gap-4 mt-2 text-[11px] text-gray-600">
              {r.inference_ms != null && (
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {r.inference_ms.toFixed(0)} ms
                </span>
              )}
              {r.device && <span>Device: {r.device.toUpperCase()}</span>}
              {r.consensus && r.consensus.models.length > 1 && (
                <span
                  className={
                    r.consensus.agreed ? "text-emerald-500" : "text-amber-500"
                  }
                >
                  {r.consensus.agreed ? "✓ Models agreed" : "⚠ Models disagreed"}
                </span>
              )}
            </div>

            {/* Warnings */}
            {r.warnings && r.warnings.length > 0 && (
              <div className="mt-2 space-y-1">
                {r.warnings.map((w, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-[11px] text-yellow-400/80 bg-yellow-900/10 border border-yellow-800/30 rounded px-2 py-1"
                  >
                    <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" />
                    {w}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {/* Failed tasks */}
      {failed.map((r) => (
        <div
          key={r.task_id}
          className="glass-card p-3 border-l-4 border-l-gray-700 flex items-center gap-3"
        >
          <XCircle className="w-4 h-4 text-gray-600 shrink-0" />
          <div>
            <p className="text-sm font-medium text-gray-400">{r.task_id}</p>
            <p className="text-xs text-gray-600">{r.error}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
