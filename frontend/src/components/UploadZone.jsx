import { useState, useRef } from 'react'

export default function UploadZone({ onFileSelect }) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDragIn = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragOut = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      validateAndSelect(files[0])
    }
  }

  const handleClick = () => inputRef.current?.click()

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSelect(e.target.files[0])
    }
  }

  const validateAndSelect = (file) => {
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg']
    if (!validTypes.includes(file.type)) {
      alert('Please upload a JPG or PNG image.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be under 10 MB.')
      return
    }
    onFileSelect(file)
  }

  return (
    <div
      onClick={handleClick}
      onDragOver={handleDrag}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDrop={handleDrop}
      className={`
        glass rounded-2xl p-12 text-center cursor-pointer
        transition-all duration-300 group
        ${isDragging
          ? 'border-purple-400/60 bg-purple-500/10 scale-[1.02]'
          : 'hover:bg-white/[0.08] hover:border-white/20'
        }
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/jpg"
        onChange={handleInputChange}
        className="hidden"
      />

      <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-purple-500/20 to-indigo-500/20 border border-purple-400/20 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
        <svg className="w-8 h-8 text-purple-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
        </svg>
      </div>

      <h3 className="text-xl font-semibold text-white/90 mb-2">
        {isDragging ? 'Drop your image here' : 'Upload Skin Image'}
      </h3>
      <p className="text-sm text-white/50 mb-4">
        Drag & drop or click to browse
      </p>
      <div className="flex items-center justify-center gap-3 text-xs text-white/30">
        <span className="px-2 py-1 rounded-md bg-white/5">JPG</span>
        <span className="px-2 py-1 rounded-md bg-white/5">PNG</span>
        <span className="text-white/20">|</span>
        <span>Max 10 MB</span>
      </div>
    </div>
  )
}
