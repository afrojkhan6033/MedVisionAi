/* ── Left sidebar — task selection ── */
import { CheckCircle2, Circle, XCircle, Zap } from "lucide-react";
import type { CatalogTask } from "../api/types";

interface Props {
  tasks: CatalogTask[];
  selectedTaskId: string | null;
  onSelect: (taskId: string | null) => void;
  onRunPanel: () => void;
  hasFile: boolean;
  isAnalyzing: boolean;
}

export default function TaskSidebar({
  tasks,
  selectedTaskId,
  onSelect,
  onRunPanel,
  hasFile,
  isAnalyzing,
}: Props) {
  return (
    <aside className="w-72 shrink-0 flex flex-col gap-3 overflow-y-auto pr-1">
      <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-1">
        Available Analyses
      </h2>

      {tasks.length === 0 && (
        <p className="text-sm text-gray-600 px-1">
          No models registered for this modality.
        </p>
      )}

      {tasks.map((task) => {
        const selected = selectedTaskId === task.task_id;
        return (
          <button
            key={task.task_id}
            onClick={() => onSelect(selected ? null : task.task_id)}
            className={`group text-left w-full px-4 py-3 rounded-xl border transition-all duration-200 ${
              selected
                ? "border-clinical-500/50 bg-clinical-950/40 shadow-lg shadow-clinical-500/5"
                : "border-gray-800 bg-gray-900/40 hover:border-gray-700 hover:bg-gray-900/60"
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-semibold text-gray-100 truncate">
                  {task.display_name}
                </h3>
                <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{task.description}</p>
              </div>
              {task.available ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
              ) : (
                <XCircle className="w-4 h-4 text-gray-600 shrink-0 mt-0.5" />
              )}
            </div>

            {/* Model badges */}
            <div className="flex flex-wrap gap-1.5 mt-2.5">
              {task.models.map((m) => (
                <span
                  key={m.id}
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium border ${
                    m.available
                      ? "border-emerald-800/50 bg-emerald-900/20 text-emerald-400"
                      : "border-gray-800 bg-gray-900/40 text-gray-600"
                  }`}
                >
                  <Circle
                    className={`w-1.5 h-1.5 ${
                      m.available ? "fill-emerald-500 text-emerald-500" : "fill-gray-700 text-gray-700"
                    }`}
                  />
                  {m.display_name}
                </span>
              ))}
            </div>

            {/* Classes */}
            <div className="text-[10px] text-gray-600 mt-2">
              {task.classes.join(" · ")}
            </div>
          </button>
        );
      })}

      {/* Panel button */}
      <button
        disabled={!hasFile || isAnalyzing || tasks.filter((t) => t.available).length === 0}
        onClick={onRunPanel}
        className="mt-auto flex items-center justify-center gap-2 w-full px-4 py-3 rounded-xl
          bg-gradient-to-r from-clinical-600 to-clinical-700 text-white font-semibold text-sm
          hover:from-clinical-500 hover:to-clinical-600 transition-all duration-200
          disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:from-clinical-600"
      >
        <Zap className="w-4 h-4" />
        Run All Available
      </button>
    </aside>
  );
}
