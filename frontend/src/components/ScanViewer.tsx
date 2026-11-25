/* ── Scan viewer with original / heatmap toggle ── */
import { useState } from "react";
import { Eye, Layers } from "lucide-react";

interface Props {
  originalSrc: string;
  overlaySrc: string | null;
  xaiType: string;
}

type ViewMode = "original" | "heatmap" | "side-by-side";

export default function ScanViewer({ originalSrc, overlaySrc, xaiType }: Props) {
  const [mode, setMode] = useState<ViewMode>(overlaySrc ? "side-by-side" : "original");
  const [opacity, setOpacity] = useState(0.5);

  const modes: { key: ViewMode; label: string }[] = [
    { key: "original", label: "Original" },
    ...(overlaySrc
      ? [
          { key: "heatmap" as ViewMode, label: xaiType === "roi_overlay" ? "ROI" : "Heatmap" },
          { key: "side-by-side" as ViewMode, label: "Compare" },
        ]
      : []),
  ];

  return (
    <div className="glass-card p-4 animate-fade-in">
      {/* Mode toggles */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1 bg-gray-900/50 rounded-lg p-0.5">
          {modes.map((m) => (
            <button
              key={m.key}
              onClick={() => setMode(m.key)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                mode === m.key
                  ? "bg-clinical-600/20 text-clinical-400"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        {mode === "heatmap" && overlaySrc && (
          <div className="flex items-center gap-2">
            <Eye className="w-3.5 h-3.5 text-gray-500" />
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-20 accent-clinical-500"
            />
          </div>
        )}
      </div>

      {/* Images */}
      <div
        className={`rounded-lg overflow-hidden bg-black/40 ${
          mode === "side-by-side" ? "grid grid-cols-2 gap-1" : "flex items-center justify-center"
        }`}
        style={{ maxHeight: "400px" }}
      >
        {mode === "original" && (
          <img src={originalSrc} alt="Original scan" className="max-h-[400px] object-contain mx-auto" />
        )}
        {mode === "heatmap" && overlaySrc && (
          <div className="relative flex items-center justify-center">
            <img src={originalSrc} alt="Original" className="max-h-[400px] object-contain" />
            <img
              src={overlaySrc}
              alt="Heatmap overlay"
              className="absolute inset-0 w-full h-full object-contain"
              style={{ opacity }}
            />
          </div>
        )}
        {mode === "side-by-side" && (
          <>
            <div className="flex items-center justify-center p-2">
              <div>
                <img src={originalSrc} alt="Original" className="max-h-[380px] object-contain rounded" />
                <p className="text-[10px] text-gray-500 text-center mt-1">Original</p>
              </div>
            </div>
            <div className="flex items-center justify-center p-2">
              {overlaySrc ? (
                <div>
                  <img src={overlaySrc} alt="Overlay" className="max-h-[380px] object-contain rounded" />
                  <p className="text-[10px] text-gray-500 text-center mt-1">
                    {xaiType === "roi_overlay" ? "ROI Overlay" : "Grad-CAM"}
                  </p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 text-gray-600">
                  <Layers className="w-8 h-8" />
                  <p className="text-xs">No overlay available</p>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
