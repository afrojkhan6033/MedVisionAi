/* ── Typed fetch wrapper for MedvisionAi API ── */
import type {
  CatalogResponse,
  HealthResponse,
  PanelResult,
  PredictionResponse,
} from "./types";

const BASE = "/api/v1";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>(`${BASE}/health`);
}

export async function fetchCatalog(): Promise<CatalogResponse> {
  return request<CatalogResponse>(`${BASE}/catalog`);
}

export async function predictSingle(
  taskId: string,
  file: File
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<PredictionResponse>(`${BASE}/predict/${taskId}`, {
    method: "POST",
    body: form,
  });
}

export async function predictPanel(
  modality: string,
  file: File
): Promise<PanelResult[]> {
  const form = new FormData();
  form.append("file", file);
  return request<PanelResult[]>(`${BASE}/predict/panel/${modality}`, {
    method: "POST",
    body: form,
  });
}
