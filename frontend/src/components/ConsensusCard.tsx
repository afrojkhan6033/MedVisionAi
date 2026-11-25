/* ── Consensus card — multi-model agreement ── */
import { CheckCircle2, XCircle, Users } from "lucide-react";
import type { ConsensusInfo } from "../api/types";

interface Props {
  consensus: ConsensusInfo;
}

export default function ConsensusCard({ consensus }: Props) {
  if (consensus.strategy === "single" && consensus.models.length <= 1) {
    return null; // Don't show for single-model tasks
  }

  return (
    <div className="glass-card p-4 animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-2">
          <Users className="w-3.5 h-3.5" />
          Model Consensus
        </h4>
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold ${
            consensus.agreed
              ? "bg-emerald-900/20 text-emerald-400 border border-emerald-800/40"
              : "bg-amber-900/20 text-amber-400 border border-amber-800/40"
          }`}
        >
          {consensus.agreed ? (
            <CheckCircle2 className="w-3 h-3" />
          ) : (
            <XCircle className="w-3 h-3" />
          )}
          {consensus.agreed ? "Agreed" : "Disagreed"}
        </div>
      </div>

      <p className="text-xs text-gray-600 mb-3">
        Strategy: <span className="text-gray-400">{consensus.strategy}</span>
      </p>

      <div className="space-y-2">
        {consensus.models.map((m, i) => (
          <div
            key={i}
            className="flex items-center justify-between bg-gray-900/40 rounded-lg px-3 py-2"
          >
            <div>
              <p className="text-sm font-medium text-gray-300">{m.name}</p>
              <p className="text-xs text-gray-500">{m.prediction}</p>
            </div>
            <span className="text-sm font-bold text-clinical-400 tabular-nums">
              {(m.confidence * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
