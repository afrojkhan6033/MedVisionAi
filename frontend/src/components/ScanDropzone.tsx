/* ── Drag-and-drop image upload zone ── */
import { useCallback, useRef, useState } from "react";
import { ImagePlus, Upload, X } from "lucide-react";

interface Props {
  file: File | null;
  filePreview: string | null;
  onFile: (f: File) => void;
  onClear: () => void;
  onAnalyze: () => void;
  canAnalyze: boolean;
  isAnalyzing: boolean;
}

export default function ScanDropzone({
  file,
  filePreview,
  onFile,
  onClear,
  onAnalyze,
  canAnalyze,
  isAnalyzing,
}: Props) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith("image/")) onFile(f);
    },
    [onFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) onFile(f);
    },
    [onFile]
  );

  if (file && filePreview) {
    return (
      <div className="glass-card p-4 animate-fade-in">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <ImagePlus className="w-4 h-4 text-clinical-400" />
            <span className="text-sm font-medium text-gray-300 truncate max-w-[200px]">
              {file.name}
            </span>
            <span className="text-xs text-gray-600">
              ({(file.size / 1024).toFixed(0)} KB)
            </span>
          </div>
          <button
            onClick={onClear}
            className="p-1 rounded-md hover:bg-gray-800 text-gray-500 hover:text-gray-300 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="relative rounded-lg overflow-hidden bg-black/40 flex items-center justify-center"
             style={{ maxHeight: "340px" }}>
          <img
            src={filePreview}
            alt="Medical scan preview"
            className="max-h-[340px] object-contain"
          />
          {isAnalyzing && (
            <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center gap-3">
              <div className="w-10 h-10 border-3 border-clinical-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-clinical-300 font-medium">Analyzing scan…</p>
            </div>
          )}
        </div>

        <button
          disabled={!canAnalyze || isAnalyzing}
          onClick={onAnalyze}
          className="mt-3 w-full py-2.5 rounded-lg bg-clinical-600 hover:bg-clinical-500 text-white
            font-semibold text-sm transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
        >
          {isAnalyzing ? "Analyzing…" : "Analyze Scan"}
        </button>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`glass-card cursor-pointer flex flex-col items-center justify-center gap-4 py-16 px-8
        transition-all duration-200 ${
          dragOver
            ? "border-clinical-500 bg-clinical-950/30 shadow-lg shadow-clinical-500/10"
            : "border-dashed hover:border-gray-600 hover:bg-gray-900/50"
        }`}
    >
      <div
        className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors ${
          dragOver ? "bg-clinical-600/20 text-clinical-400" : "bg-gray-800/60 text-gray-500"
        }`}
      >
        <Upload className="w-7 h-7" />
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-gray-300">
          Drop your medical scan here
        </p>
        <p className="text-xs text-gray-600 mt-1">
          JPEG, PNG, BMP or WebP • Max 10 MB
        </p>
      </div>
      <input ref={inputRef} type="file" accept="image/*" onChange={handleChange} className="hidden" />
    </div>
  );
}
