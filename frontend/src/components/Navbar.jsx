export default function Navbar() {
  return (
    <nav className="glass sticky top-0 z-50 px-6 py-4">
      <div className="max-w-4xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-400 to-indigo-600 flex items-center justify-center font-bold text-sm">
            SI
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">SkinID</h1>
            <p className="text-[10px] text-white/50 -mt-0.5 tracking-wide uppercase">Skin Lesion Identifier</p>
          </div>
        </div>
        <div className="text-xs text-white/40 hidden sm:block">
          AI-Powered Dermatology Screening
        </div>
      </div>
    </nav>
  )
}
