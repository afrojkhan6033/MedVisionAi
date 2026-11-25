/* ── TypeScript interfaces mirroring backend schemas ── */

export interface ModelResult {
  name: string;
  prediction: string;
  confidence: number;
}

export interface ConsensusInfo {
  agreed: boolean;
  strategy: string;
  models: ModelResult[];
}

export interface ExplainabilityResult {
  type: string;
  overlay_base64: string | null;
}

export interface PredictionResponse {
  request_id: string;
  task_id: string;
  display_name: string;
  prediction: string;
  confidence: number;
  is_abnormal: boolean;
  threshold_used: number | null;
  probabilities: Record<string, number>;
  consensus: ConsensusInfo;
  explainability: ExplainabilityResult | null;
  warnings: string[];
  inference_ms: number;
  device: string;
}

export interface ModelInfo {
  id: string;
  display_name: string;
  enabled: boolean;
  available: boolean;
}

export interface CatalogTask {
  task_id: string;
  display_name: string;
  description: string;
  modality: string;
  classes: string[];
  models: ModelInfo[];
  available: boolean;
  explainability: string;
}

export interface ModalityInfo {
  id: string;
  display_name: string;
  icon: string;
}

export interface CatalogResponse {
  tasks: CatalogTask[];
  modalities: ModalityInfo[];
}

export interface HealthResponse {
  status: string;
  device: string;
  loaded_pipelines: string[];
  gpu_name: string | null;
}

export interface PanelResult {
  task_id: string;
  error?: string;
  // All PredictionResponse fields when successful
  request_id?: string;
  display_name?: string;
  prediction?: string;
  confidence?: number;
  is_abnormal?: boolean;
  threshold_used?: number | null;
  probabilities?: Record<string, number>;
  consensus?: ConsensusInfo;
  explainability?: ExplainabilityResult | null;
  warnings?: string[];
  inference_ms?: number;
  device?: string;
}
