const BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "https://web-production-6f0c32.up.railway.app";

/**
 * Sends raw values to the backend
 *
 * @param {Object} payload - keys must match the training columns exactly
 * @returns {Promise<{ prediction: "Satisfied" | "Neutral or Dissatisfied" }>}
 */
export async function predictSatisfaction(payload) {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let detail = `Prediction request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = Array.isArray(body.detail)
        ? body.detail.map((d) => d.msg).join("; ")
        : body.detail;
    } catch {
      
    }
    throw new Error(detail);
  }

  return res.json();
}
