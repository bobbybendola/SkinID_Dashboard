import { useState } from 'react'

const BODY_LOCATIONS = [
  { value: 'anterior_torso', label: 'Anterior torso (Front of the chest/stomach)' },
  { value: 'posterior_torso', label: 'Posterior torso (Back)' },
  { value: 'upper_extremity', label: 'Upper extremity (Arms/Shoulders)' },
  { value: 'lower_extremity', label: 'Lower extremity (Legs/Feet)' },
  { value: 'head_neck', label: 'Head/neck' },
  { value: 'palms_soles', label: 'Palms/soles' },
  { value: 'lateral_torso', label: 'Lateral torso (Sides)' },
  { value: 'oral_genital', label: 'Oral/genital' },
]

export default function PatientForm({ preview, measurements, onSubmit, onBack }) {
  const [sex, setSex] = useState('')
  const [age, setAge] = useState('')
  const [location, setLocation] = useState('')
  const [errors, setErrors] = useState({})

  const validate = () => {
    const errs = {}
    if (!sex) errs.sex = 'Please select your sex'
    if (!age || age < 0 || age > 150) errs.age = 'Please enter a valid age'
    if (!location) errs.location = 'Please select a body location'
    setErrors(errs)
    return Object.keys(errs).length === 0
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!validate()) return
    onSubmit({ sex, age: parseInt(age), location })
  }

  return (
    <div className="glass rounded-2xl p-6 sm:p-8">
      <h3 className="text-lg font-semibold mb-1">Patient Information</h3>
      <p className="text-sm text-white/50 mb-6">Provide details for a more accurate assessment</p>

      <div className="flex flex-col sm:flex-row gap-6">
        {/* Image preview + measurement summary */}
        <div className="sm:w-48 flex-shrink-0">
          <div className="rounded-xl overflow-hidden border border-white/10 mb-3">
            <img src={preview} alt="Uploaded lesion" className="w-full h-auto" />
          </div>
          <div className="grid grid-cols-2 gap-2 text-center">
            <div className="bg-white/5 rounded-lg p-2">
              <p className="text-[10px] uppercase tracking-wider text-white/40">Width</p>
              <p className="text-sm font-semibold text-purple-300">
                {measurements.width_mm ? `${measurements.width_mm}mm` : `${measurements.width_px}px`}
              </p>
            </div>
            <div className="bg-white/5 rounded-lg p-2">
              <p className="text-[10px] uppercase tracking-wider text-white/40">Height</p>
              <p className="text-sm font-semibold text-emerald-300">
                {measurements.height_mm ? `${measurements.height_mm}mm` : `${measurements.height_px}px`}
              </p>
            </div>
          </div>
        </div>

        {/* Form fields */}
        <form onSubmit={handleSubmit} className="flex-1 space-y-5">
          {/* Sex */}
          <div>
            <label className="block text-sm font-medium text-white/70 mb-2">Sex</label>
            <div className="flex gap-2">
              {['male', 'female'].map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => { setSex(s); setErrors(prev => ({...prev, sex: undefined})) }}
                  className={`
                    flex-1 py-2.5 rounded-xl text-sm font-medium transition-all
                    ${sex === s
                      ? 'bg-purple-500/25 border border-purple-400/40 text-purple-200'
                      : 'bg-white/5 border border-white/10 text-white/50 hover:bg-white/10 hover:text-white/70'
                    }
                  `}
                >
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </button>
              ))}
            </div>
            {errors.sex && <p className="text-xs text-red-400 mt-1">{errors.sex}</p>}
          </div>

          {/* Age */}
          <div>
            <label className="block text-sm font-medium text-white/70 mb-2">Age</label>
            <input
              type="number"
              value={age}
              onChange={(e) => { setAge(e.target.value); setErrors(prev => ({...prev, age: undefined})) }}
              placeholder="Enter your age"
              min="0"
              max="150"
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-white/30 outline-none focus:border-purple-400/50 focus:bg-white/[0.07] transition-all"
            />
            {errors.age && <p className="text-xs text-red-400 mt-1">{errors.age}</p>}
          </div>

          {/* Body Location */}
          <div>
            <label className="block text-sm font-medium text-white/70 mb-2">Lesion Location</label>
            <select
              value={location}
              onChange={(e) => { setLocation(e.target.value); setErrors(prev => ({...prev, location: undefined})) }}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white outline-none focus:border-purple-400/50 focus:bg-white/[0.07] transition-all appearance-none"
              style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='rgba(255,255,255,0.4)' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M19 9l-7 7-7-7'/%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 12px center', backgroundSize: '18px' }}
            >
              <option value="" className="bg-slate-900">Select location</option>
              {BODY_LOCATIONS.map(loc => (
                <option key={loc.value} value={loc.value} className="bg-slate-900">
                  {loc.label}
                </option>
              ))}
            </select>
            {errors.location && <p className="text-xs text-red-400 mt-1">{errors.location}</p>}
          </div>

          <div className="flex items-center gap-3 pt-2">
            <button
              type="button"
              onClick={onBack}
              className="px-5 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white/60 hover:text-white hover:bg-white/10 transition-all"
            >
              Back
            </button>
            <div className="flex-1" />
            <button
              type="submit"
              className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 text-sm font-medium hover:from-purple-500 hover:to-indigo-500 transition-all shadow-lg shadow-purple-900/30"
            >
              Analyze
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
