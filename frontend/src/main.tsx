import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import type { PolyphonyData } from "./types";
import "./styles.css";

// The React app fetches its data from the Python backend at /api/data.
// This keeps the built HTML fully static (no server-side string substitution)
// and means the data can be reloaded without regenerating the bundle.
function Bootstrap() {
  const [data, setData] = useState<PolyphonyData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/data")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<PolyphonyData>;
      })
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <div style={{ padding: 40, color: "#ffa657", fontFamily: "system-ui" }}>
        <h2>Failed to load polyphony data</h2>
        <p>GET /api/data failed: {error}</p>
        <p>Is the Python server running? Try: `polyphony serve &lt;audio&gt;`</p>
      </div>
    );
  }
  if (!data) {
    return (
      <div style={{ padding: 40, color: "#8b949e", fontFamily: "system-ui" }}>Loading…</div>
    );
  }
  return <App data={data} />;
}

const container = document.getElementById("app");
if (!container) throw new Error("polyphony: #app root element missing");

createRoot(container).render(
  <StrictMode>
    <Bootstrap />
  </StrictMode>,
);
