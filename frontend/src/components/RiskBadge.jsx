const RISK_CONFIG = {
  low: {
    color: 'emerald',
    bg: 'bg-emerald-500/15',
    border: 'border-emerald-400/30',
    text: 'text-emerald-300',
    label: 'LOW RISK',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
  moderate: {
    color: 'amber',
    bg: 'bg-amber-500/15',
    border: 'border-amber-400/30',
    text: 'text-amber-300',
    label: 'MODERATE RISK',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z" />
      </svg>
    ),
  },
  high: {
    color: 'red',
    bg: 'bg-red-500/15',
    border: 'border-red-400/30',
    text: 'text-red-300',
    label: 'HIGH RISK',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z" />
      </svg>
    ),
  },
}

export default function RiskBadge({ riskLevel, probability }) {
  const config = RISK_CONFIG[riskLevel] || RISK_CONFIG.low

  return (
    <div className={`inline-flex items-center gap-3 px-4 py-2.5 rounded-xl ${config.bg} border ${config.border}`}>
      <span className={config.text}>{config.icon}</span>
      <div>
        <span className={`text-sm font-bold ${config.text}`}>{config.label}</span>
        <span className="text-xs text-white/40 ml-2">({(probability * 100).toFixed(1)}%)</span>
      </div>
    </div>
  )
}
