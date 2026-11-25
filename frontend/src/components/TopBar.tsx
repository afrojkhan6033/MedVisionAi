/* ── Top navigation bar ── */
import { Activity, Brain, Stethoscope, Wifi, WifiOff } from "lucide-react";
import type { HealthResponse, ModalityInfo } from "../api/types";

interface Props {
  modalities: ModalityInfo[];
  selectedModality: string;
  onSelectModality: (id: string) => void;
  health: HealthResponse | null;
}

const ICONS: Record<string, React.ReactNode> = {
  brain: <Brain className="w-4 h-4" />,
  stethoscope: <Stethoscope className="w-4 h-4" />,
};

export default function TopBar({ modalities, selectedModality, onSelectModality, health }: Props) {
  const online = health?.status === "ok";

  return (
    <header className="sticky top-0 z-50 glass-card border-b border-gray-800 px-6 py-3">
      <div className="max-w-[1600px] mx-auto flex items-center justify-between gap-4">
        {/* Logo */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-clinical-500 to-clinical-700 flex items-center justify-center">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-clinical-400 to-clinical-600 bg-clip-text text-transparent">
              MedvisionAi
            </h1>
            <p className="text-[10px] text-gray-500 -mt-0.5 tracking-wide uppercase">
              Unified Medical Imaging
            </p>
          </div>
        </div>

        {/* Modality tabs */}
        <nav className="flex items-center gap-1 bg-gray-900/50 rounded-lg p-1">
          {modalities.map((m) => (
            <button
              key={m.id}
              onClick={() => onSelectModality(m.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200
                ${
                  selectedModality === m.id
                    ? "bg-clinical-600/20 text-clinical-400 shadow-sm shadow-clinical-500/10"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/50"
                }`}
            >
              {ICONS[m.icon] ?? <Activity className="w-4 h-4" />}
              {m.display_name}
            </button>
          ))}
        </nav>

        {/* Health pill */}
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border shrink-0
            ${
              online
                ? "border-emerald-700/50 bg-emerald-900/20 text-emerald-400"
                : "border-red-700/50 bg-red-900/20 text-red-400"
            }`}
        >
          {online ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
          {online ? `API Online • ${health?.device?.toUpperCase()}` : "API Offline"}
        </div>
      </div>
    </header>
  );
}
