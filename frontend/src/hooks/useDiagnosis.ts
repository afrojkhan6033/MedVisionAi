/* ── Diagnosis state machine hook ── */
import { useCallback, useEffect, useReducer } from "react";
import type {
  CatalogResponse,
  CatalogTask,
  HealthResponse,
  PanelResult,
  PredictionResponse,
} from "../api/types";
import { fetchCatalog, fetchHealth, predictPanel, predictSingle } from "../api/client";

export type Status = "idle" | "loading" | "analyzing" | "done" | "error";

interface State {
  catalog: CatalogResponse | null;
  health: HealthResponse | null;
  selectedModality: string;
  selectedTaskId: string | null;
  file: File | null;
  filePreview: string | null;
  status: Status;
  result: PredictionResponse | null;
  panelResults: PanelResult[] | null;
  error: string | null;
}

type Action =
  | { type: "INIT_START" }
  | { type: "SET_CATALOG"; catalog: CatalogResponse }
  | { type: "SET_HEALTH"; health: HealthResponse }
  | { type: "SELECT_MODALITY"; modality: string }
  | { type: "SELECT_TASK"; taskId: string | null }
  | { type: "SET_FILE"; file: File; preview: string }
  | { type: "CLEAR_FILE" }
  | { type: "ANALYZE_START" }
  | { type: "ANALYZE_SINGLE_OK"; result: PredictionResponse }
  | { type: "ANALYZE_PANEL_OK"; results: PanelResult[] }
  | { type: "ANALYZE_FAIL"; error: string }
  | { type: "RESET" };

const initial: State = {
  catalog: null,
  health: null,
  selectedModality: "brain_mri",
  selectedTaskId: null,
  file: null,
  filePreview: null,
  status: "loading",
  result: null,
  panelResults: null,
  error: null,
};

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case "INIT_START":
      return { ...s, status: "loading" };
    case "SET_CATALOG":
      return { ...s, catalog: a.catalog, status: s.status === "loading" ? "idle" : s.status };
    case "SET_HEALTH":
      return { ...s, health: a.health };
    case "SELECT_MODALITY":
      return { ...s, selectedModality: a.modality, selectedTaskId: null, result: null, panelResults: null, error: null };
    case "SELECT_TASK":
      return { ...s, selectedTaskId: a.taskId, result: null, panelResults: null, error: null };
    case "SET_FILE":
      return { ...s, file: a.file, filePreview: a.preview, result: null, panelResults: null, error: null };
    case "CLEAR_FILE":
      return { ...s, file: null, filePreview: null, result: null, panelResults: null, error: null };
    case "ANALYZE_START":
      return { ...s, status: "analyzing", result: null, panelResults: null, error: null };
    case "ANALYZE_SINGLE_OK":
      return { ...s, status: "done", result: a.result };
    case "ANALYZE_PANEL_OK":
      return { ...s, status: "done", panelResults: a.results };
    case "ANALYZE_FAIL":
      return { ...s, status: "error", error: a.error };
    case "RESET":
      return { ...initial, catalog: s.catalog, health: s.health, status: "idle" };
    default:
      return s;
  }
}

export function useDiagnosis() {
  const [state, dispatch] = useReducer(reducer, initial);

  // Fetch catalog + health on mount
  useEffect(() => {
    dispatch({ type: "INIT_START" });
    fetchCatalog()
      .then((c) => dispatch({ type: "SET_CATALOG", catalog: c }))
      .catch((err) => dispatch({ type: "ANALYZE_FAIL", error: err.message || "Failed to connect to API" }));
    fetchHealth()
      .then((h) => dispatch({ type: "SET_HEALTH", health: h }))
      .catch(() => {});
  }, []);

  const selectModality = useCallback((m: string) => dispatch({ type: "SELECT_MODALITY", modality: m }), []);
  const selectTask = useCallback((t: string | null) => dispatch({ type: "SELECT_TASK", taskId: t }), []);

  const setFile = useCallback((f: File) => {
    const url = URL.createObjectURL(f);
    dispatch({ type: "SET_FILE", file: f, preview: url });
  }, []);
  const clearFile = useCallback(() => dispatch({ type: "CLEAR_FILE" }), []);

  const analyzeSingle = useCallback(async () => {
    if (!state.file || !state.selectedTaskId) return;
    dispatch({ type: "ANALYZE_START" });
    try {
      const r = await predictSingle(state.selectedTaskId, state.file);
      dispatch({ type: "ANALYZE_SINGLE_OK", result: r });
    } catch (e: any) {
      dispatch({ type: "ANALYZE_FAIL", error: e.message ?? "Analysis failed" });
    }
  }, [state.file, state.selectedTaskId]);

  const analyzePanel = useCallback(async () => {
    if (!state.file) return;
    dispatch({ type: "ANALYZE_START" });
    try {
      const r = await predictPanel(state.selectedModality, state.file);
      dispatch({ type: "ANALYZE_PANEL_OK", results: r });
    } catch (e: any) {
      dispatch({ type: "ANALYZE_FAIL", error: e.message ?? "Panel analysis failed" });
    }
  }, [state.file, state.selectedModality]);

  const reset = useCallback(() => dispatch({ type: "RESET" }), []);

  // Filtered tasks for selected modality
  const modalityTasks: CatalogTask[] =
    state.catalog?.tasks.filter((t) => t.modality === state.selectedModality) ?? [];

  return {
    ...state,
    modalityTasks,
    selectModality,
    selectTask,
    setFile,
    clearFile,
    analyzeSingle,
    analyzePanel,
    reset,
  };
}
