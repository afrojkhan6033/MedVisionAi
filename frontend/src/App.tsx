/* ── App — main 3-zone clinical workspace ── */
import { useDiagnosis } from "./hooks/useDiagnosis";
import TopBar from "./components/TopBar";
import TaskSidebar from "./components/TaskSidebar";
import ScanDropzone from "./components/ScanDropzone";
import ScanViewer from "./components/ScanViewer";
import VerdictCard from "./components/VerdictCard";
import ProbBars from "./components/ProbBars";
import ConsensusCard from "./components/ConsensusCard";
import PanelResults from "./components/PanelResults";
import DisclaimerBar from "./components/DisclaimerBar";
import { AlertCircle, Loader2, Brain, Stethoscope, Activity } from "lucide-react";

export default function App() {
  const {
    catalog,
    health,
    selectedModality,
    selectedTaskId,
    file,
    filePreview,
    status,
    result,
    panelResults,
    error,
    modalityTasks,
    selectModality,
    selectTask,
    setFile,
    clearFile,
    analyzeSingle,
    analyzePanel,
    reset,
  } = useDiagnosis();

  const modalities = catalog?.modalities ?? [
    { id: "brain_mri", display_name: "Brain MRI", icon: "brain" },
    { id: "chest_scan", display_name: "Chest Scan", icon: "stethoscope" },
  ];

  const isAnalyzing = status === "analyzing";
  const hasResult = status === "done" && (result !== null || panelResults !== null);

  // Loading state — waiting for catalog
  if (!catalog) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-clinical-500 to-clinical-700 flex items-center justify-center">
          <Activity className="w-6 h-6 text-white animate-pulse" />
        </div>
        <div className="text-center">
          <h1 className="text-lg font-bold text-gray-200">MedvisionAi</h1>
          <p className="text-sm text-gray-500 mt-1 flex items-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            Connecting to API…
          </p>
        </div>
        <p className="text-xs text-gray-700 mt-4 max-w-sm text-center">
          Make sure the backend is running:{" "}
          <code className="text-gray-500">python -m backend.main</code>
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col pb-10">
      <TopBar
        modalities={modalities}
        selectedModality={selectedModality}
        onSelectModality={selectModality}
        health={health}
      />

      <main className="flex-1 max-w-[1600px] mx-auto w-full px-6 py-6">
        <div className="flex gap-6 items-start">
          {/* ── Zone 1: Left sidebar — task selection ── */}
          <TaskSidebar
            tasks={modalityTasks}
            selectedTaskId={selectedTaskId}
            onSelect={selectTask}
            onRunPanel={analyzePanel}
            hasFile={file !== null}
            isAnalyzing={isAnalyzing}
          />

          {/* ── Zone 2: Center — scan upload & viewer ── */}
          <div className="flex-1 min-w-0 space-y-4">
            {/* Upload / preview */}
            {!hasResult ? (
              <ScanDropzone
                file={file}
                filePreview={filePreview}
                onFile={setFile}
                onClear={clearFile}
                onAnalyze={analyzeSingle}
                canAnalyze={selectedTaskId !== null}
                isAnalyzing={isAnalyzing}
              />
            ) : result?.explainability?.overlay_base64 || filePreview ? (
              <ScanViewer
                originalSrc={filePreview!}
                overlaySrc={result?.explainability?.overlay_base64 ?? null}
                xaiType={result?.explainability?.type ?? "gradcam"}
              />
            ) : null}

            {/* Error */}
            {status === "error" && error && (
              <div className="glass-card p-4 border-red-700/40 bg-red-950/20 flex items-start gap-3 animate-fade-in">
                <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-red-300">Analysis Failed</p>
                  <p className="text-xs text-red-400/80 mt-1">{error}</p>
                </div>
              </div>
            )}

            {/* Empty state hint */}
            {status === "idle" && !file && (
              <div className="glass-card p-8 flex flex-col items-center gap-4 text-center">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-xl bg-gray-800/60 flex items-center justify-center">
                    {selectedModality === "brain_mri" ? (
                      <Brain className="w-6 h-6 text-gray-600" />
                    ) : (
                      <Stethoscope className="w-6 h-6 text-gray-600" />
                    )}
                  </div>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-gray-400">
                    Ready to Analyze
                  </h3>
                  <p className="text-xs text-gray-600 mt-1 max-w-sm">
                    Select an analysis task from the sidebar, then upload a medical scan
                    to get AI-powered predictions with explainability overlays.
                  </p>
                </div>
              </div>
            )}

            {/* New Analysis button when done */}
            {hasResult && (
              <button
                onClick={reset}
                className="w-full py-2 rounded-lg border border-gray-800 text-sm text-gray-400
                  hover:bg-gray-900/60 hover:text-gray-200 transition-colors"
              >
                ← New Analysis
              </button>
            )}
          </div>

          {/* ── Zone 3: Right — results panel ── */}
          <div className="w-80 shrink-0 space-y-4 overflow-y-auto">
            {/* Single result */}
            {result && (
              <>
                <VerdictCard result={result} />
                <ProbBars
                  probabilities={result.probabilities}
                  prediction={result.prediction}
                  threshold={result.threshold_used}
                />
                <ConsensusCard consensus={result.consensus} />
              </>
            )}

            {/* Panel results */}
            {panelResults && <PanelResults results={panelResults} />}

            {/* Placeholder when no results */}
            {!hasResult && !isAnalyzing && (
              <div className="glass-card p-6 flex flex-col items-center gap-3 text-center">
                <div className="w-10 h-10 rounded-xl bg-gray-800/40 flex items-center justify-center">
                  <Activity className="w-5 h-5 text-gray-700" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-gray-500">Results Panel</p>
                  <p className="text-[11px] text-gray-700 mt-1">
                    Predictions, confidence scores, and explainability overlays will
                    appear here after analysis.
                  </p>
                </div>
              </div>
            )}

            {/* Analyzing placeholder */}
            {isAnalyzing && (
              <div className="glass-card p-6 flex flex-col items-center gap-3 animate-shimmer">
                <Loader2 className="w-8 h-8 text-clinical-500 animate-spin" />
                <p className="text-xs text-gray-500">Running inference…</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <DisclaimerBar />
    </div>
  );
}
