export default function Hero() {
  return (
    <section className="text-center py-12 sm:py-16">
      <h2 className="text-4xl sm:text-5xl font-bold tracking-tight bg-gradient-to-r from-white via-purple-200 to-indigo-300 bg-clip-text text-transparent">
        Understand Your Skin
      </h2>
      <p className="mt-4 text-base sm:text-lg text-white/60 max-w-xl mx-auto leading-relaxed">
        Upload a photo of your skin lesion. Our AI will measure it, assess risk, and identify what it might be.
      </p>
      <div className="flex items-center justify-center gap-6 mt-6 text-xs text-white/40">
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
          Upload Image
        </span>
        <span className="text-white/20">&#8594;</span>
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
          Measure
        </span>
        <span className="text-white/20">&#8594;</span>
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
          Get Results
        </span>
      </div>
    </section>
  )
}
