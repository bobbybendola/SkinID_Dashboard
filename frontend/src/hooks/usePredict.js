import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000'

export default function usePredict() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // ⚡ Updated predict to send JSON instead of FormData
  const predict = async (file, measurements, patientInfo) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // ✅ Build JSON exactly like Flask expects
      const payload = {
        sex: patientInfo.sex,
        age_approx: Number(patientInfo.age),
        anatom_site_general_challenge: patientInfo.location || "unknown",
        width: measurements.width_mm ?? measurements.width_px,
        height: measurements.height_mm ?? measurements.height_px
      }

      // ⚡ Call backend with JSON
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => null)
        throw new Error(errData?.detail || `Server error: ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to connect to the server')
    } finally {
      setLoading(false)
    }
  }

  return { result, loading, error, predict }
}