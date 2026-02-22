import RiskBadge from './RiskBadge'

export default function ResultsCard({ result, loading, error, preview, measurements, patientInfo, onReset }) {
  if (loading) {
    return (
      <div className="glass rounded-2xl p-12 text-center">
        <div className="w-16 h-16 mx-auto mb-6 rounded-full border-2 border-purple-400/30 border-t-purple-400 animate-spin" />
        <h3 className="text-lg font-semibold text-white/80 mb-2">Analyzing your image...</h3>
        <p className="text-sm text-white/40">This may take a few moments</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="glass rounded-2xl p-8 text-center">
        <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-red-500/15 border border-red-400/20 flex items-center justify-center">
          <svg className="w-7 h-7 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-red-300 mb-2">Analysis Failed</h3>
        <p className="text-sm text-white/50 mb-6">{error}</p>
        <button onClick={onReset} className="px-5 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white/60 hover:text-white hover:bg-white/10 transition-all">
          Try Again
        </button>
      </div>
    )
  }

  if (!result) return null

  // ✅ Map prediction number to text
  const predictionText = result.prediction === 1 ? "Malignant" : "Benign"
  const confidencePercent = (result.confidence * 100).toFixed(2)

  return (
    <div className="space-y-4">
      <div className="glass-strong rounded-2xl p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row gap-6">
          {/* Image + info sidebar */}
          <div className="sm:w-52 flex-shrink-0 space-y-3">
            {preview && (
              <div className="rounded-xl overflow-hidden border border-white/10">
                <img src={preview} alt="Analyzed lesion" className="w-full h-auto" />
              </div>
            )}
            <div className="grid grid-cols-2 gap-2 text-center text-xs">
              <div className="bg-white/5 rounded-lg p-2">
                <span className="text-white/40">Width</span>
                <p className="font-semibold text-purple-300">{measurements.width_mm ? `${measurements.width_mm}mm` : `${measurements.width_px}px`}</p>
              </div>
              <div className="bg-white/5 rounded-lg p-2">
                <span className="text-white/40">Height</span>
                <p className="font-semibold text-emerald-300">{measurements.height_mm ? `${measurements.height_mm}mm` : `${measurements.height_px}px`}</p>
              </div>
            </div>
            <div className="bg-white/5 rounded-lg p-2 text-xs text-center space-y-1">
              <div><span className="text-white/40">Sex:</span> <span className="text-white/70 capitalize">{patientInfo.sex}</span></div>
              <div><span className="text-white/40">Age:</span> <span className="text-white/70">{patientInfo.age}</span></div>
              <div><span className="text-white/40">Location:</span> <span className="text-white/70 capitalize">{patientInfo.location.replace(/_/g, ' ')}</span></div>
            </div>
          </div>

          {/* Results */}
          <div className="flex-1 space-y-5">
            <div className="glass rounded-xl p-4 text-center">
              <h3 className="text-xl font-bold mb-2">Prediction</h3>
              <p className="text-2xl font-semibold text-purple-300">{predictionText}</p>
              <p className="text-white/60 mt-1">Confidence: {confidencePercent}%</p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-center">
        <button
          onClick={onReset}
          className="px-6 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white/60 hover:text-white hover:bg-white/10 transition-all"
        >
          Analyze Another Image
        </button>
      </div>
    </div>
  )
}