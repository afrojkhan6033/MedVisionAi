/* ── Probability bars — class-wise breakdown ── */

interface Props {
  probabilities: Record<string, number>;
  prediction: string;
  threshold?: number | null;
}

export default function ProbBars({ probabilities, prediction, threshold }: Props) {
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...sorted.map(([, v]) => v), 0.01);

  return (
    <div className="glass-card p-4 animate-fade-in">
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
        Class Probabilities
      </h4>
      <div className="space-y-2.5">
        {sorted.map(([cls, prob]) => {
          const pct = prob * 100;
          const isPred = cls === prediction;
          return (
            <div key={cls}>
              <div className="flex items-center justify-between mb-1">
                <span
                  className={`text-xs font-medium ${
                    isPred ? "text-clinical-400" : "text-gray-400"
                  }`}
                >
                  {cls}
                  {isPred && (
                    <span className="ml-1.5 text-[10px] text-clinical-500 font-semibold">
                      ● PREDICTED
                    </span>
                  )}
                </span>
                <span
                  className={`text-xs font-bold tabular-nums ${
                    isPred ? "text-clinical-300" : "text-gray-500"
                  }`}
                >
                  {pct.toFixed(1)}%
                </span>
              </div>
              <div className="relative h-2 rounded-full bg-gray-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out ${
                    isPred
                      ? "bg-gradient-to-r from-clinical-600 to-clinical-400"
                      : "bg-gray-700"
                  }`}
                  style={{ width: `${(prob / max) * 100}%` }}
                />
                {threshold != null && (
                  <div
                    className="absolute top-0 bottom-0 w-px bg-yellow-500/60"
                    style={{ left: `${(threshold / max) * 100}%` }}
                    title={`Threshold: ${(threshold * 100).toFixed(0)}%`}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
