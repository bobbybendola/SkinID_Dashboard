export default function Disclaimer() {
  return (
    <div className="mt-12 text-center px-4">
      <div className="inline-flex items-start gap-2 max-w-2xl text-xs text-white/30 leading-relaxed">
        <svg className="w-4 h-4 mt-0.5 flex-shrink-0 text-white/20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
        </svg>
        <p>
          SkinID is an informational tool and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a board-certified dermatologist for any skin concerns.
        </p>
      </div>
    </div>
  )
}
