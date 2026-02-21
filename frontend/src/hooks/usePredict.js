import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function usePredict() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const predict = async (file, measurements, patientInfo) => {
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('width_px', measurements.width_px)
    formData.append('height_px', measurements.height_px)
    if (measurements.width_mm) formData.append('width_mm', measurements.width_mm)
    if (measurements.height_mm) formData.append('height_mm', measurements.height_mm)
    formData.append('sex', patientInfo.sex)
    formData.append('age', patientInfo.age)
    formData.append('anatom_site', patientInfo.location)

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
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
