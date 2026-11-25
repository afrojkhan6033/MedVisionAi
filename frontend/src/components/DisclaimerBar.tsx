/* ── Disclaimer bar — fixed footer ── */
import { ShieldAlert } from "lucide-react";

export default function DisclaimerBar() {
  return (
    <footer className="fixed bottom-0 inset-x-0 z-40 bg-gray-950/90 backdrop-blur border-t border-gray-900 px-6 py-2">
      <div className="max-w-[1600px] mx-auto flex items-center gap-2 text-[11px] text-gray-600">
        <ShieldAlert className="w-3.5 h-3.5 shrink-0 text-yellow-600/60" />
        <p>
          <strong className="text-gray-500">Research Tool Only.</strong> MedvisionAi is not a medical device.
          Predictions are AI-generated and must not be used for clinical diagnosis or treatment decisions.
          Always consult a qualified healthcare professional.
        </p>
      </div>
    </footer>
  );
}
