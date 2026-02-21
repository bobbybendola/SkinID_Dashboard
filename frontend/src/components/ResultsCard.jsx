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

  const { malignancy, identity, top_3 } = result

  return (
    <div className="space-y-4">
      {/* Main result card */}
      <div className="glass-strong rounded-2xl p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row gap-6">
          {/* Image + info sidebar */}
          <div className="sm:w-52 flex-shrink-0 space-y-3">
            <div className="rounded-xl overflow-hidden border border-white/10">
              <img src={preview} alt="Analyzed lesion" className="w-full h-auto" />
            </div>
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
            {/* Identity */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg bg-indigo-500/20 flex items-center justify-center">
                  <svg className="w-4 h-4 text-indigo-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
                  </svg>
                </div>
                <h4 className="text-sm font-medium text-white/50 uppercase tracking-wider">Identification</h4>
              </div>
              <h3 className="text-xl font-bold mb-1">{identity.name}</h3>
              <div className="flex items-center gap-2 mb-3">
                <div className="h-1.5 flex-1 max-w-32 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-indigo-400 rounded-full transition-all duration-1000"
                    style={{ width: `${(identity.confidence * 100).toFixed(0)}%` }}
                  />
                </div>
                <span className="text-sm font-semibold text-purple-300">{(identity.confidence * 100).toFixed(1)}%</span>
              </div>
              <p className="text-sm text-white/60 leading-relaxed">{identity.description}</p>
              {identity.clinical_note && (
                <p className="mt-2 text-xs text-white/30 italic">{identity.clinical_note}</p>
              )}
            </div>

            {/* Risk assessment */}
            <div className="glass rounded-xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
                  <svg className="w-4 h-4 text-white/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z" />
                  </svg>
                </div>
                <h4 className="text-sm font-medium text-white/50 uppercase tracking-wider">Risk Assessment</h4>
              </div>
              <RiskBadge riskLevel={malignancy.risk_level} probability={malignancy.probability} />
              <p className="text-sm text-white/60 mt-3">{malignancy.recommendation}</p>
            </div>

            {/* Top 3 */}
            {top_3 && top_3.length > 1 && (
              <div>
                <h4 className="text-xs font-medium text-white/40 uppercase tracking-wider mb-2">Other Possibilities</h4>
                <div className="space-y-2">
                  {top_3.slice(1).map((item, i) => (
                    <div key={i} className="flex items-center gap-3 bg-white/[0.03] rounded-lg p-2.5">
                      <div className="flex-1">
                        <span className="text-sm text-white/70">{item.name}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-white/25 rounded-full"
                            style={{ width: `${(item.confidence * 100).toFixed(0)}%` }}
                          />
                        </div>
                        <span className="text-xs text-white/40 w-10 text-right">{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
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
