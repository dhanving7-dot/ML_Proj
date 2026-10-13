import { useId, type ReactNode } from "react";
import { cn } from "../../lib/format";

const control =
  "w-full rounded-xl border bg-surface-2/80 px-3.5 py-2.5 text-sm text-ink placeholder:text-ink-3 outline-none transition-[border-color,box-shadow] duration-300 focus:border-violet focus:shadow-[0_0_0_4px_rgb(139_92_246/0.18)]";

interface Base {
  label: string;
  error?: string;
  hint?: ReactNode;
  required?: boolean;
  className?: string;
}

export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  error,
  hint,
  required,
  placeholder,
  className,
  name,
}: Base & { value: string; onChange: (v: string) => void; min?: number; max?: number; step?: number | "any"; placeholder?: string; name?: string }) {
  const id = useId();
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <label htmlFor={id} className="flex items-center justify-between gap-2 text-[13px] font-medium text-ink-2">
        <span>
          {label}
          {required && <span className="text-ink-3"> *</span>}
        </span>
        {min != null && max != null && <span className="font-mono text-[10px] text-ink-3">{`${min}–${max}`}</span>}
      </label>
      <input
        id={id}
        name={name}
        type="number"
        inputMode="decimal"
        value={value}
        min={min}
        max={max}
        step={step ?? "any"}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        aria-invalid={!!error}
        aria-describedby={error ? `${id}-err` : hint ? `${id}-hint` : undefined}
        className={cn(control, "tabular", error ? "border-critical/70" : "border-line")}
      />
      {error ? (
        <p id={`${id}-err`} className="text-xs text-critical-text">
          {error}
        </p>
      ) : hint ? (
        <p id={`${id}-hint`} className="text-xs text-ink-3">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

export function SelectField({
  label,
  value,
  onChange,
  options,
  error,
  hint,
  required,
  className,
  name,
}: Base & { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; name?: string }) {
  const id = useId();
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <label htmlFor={id} className="text-[13px] font-medium text-ink-2">
        {label}
        {required && <span className="text-ink-3"> *</span>}
      </label>
      <select
        id={id}
        name={name}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-invalid={!!error}
        className={cn(control, "cursor-pointer appearance-none bg-[url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%237d83a7' stroke-width='2'><path d='m6 9 6 6 6-6'/></svg>\")] bg-[right_0.9rem_center] bg-no-repeat pr-9", error ? "border-critical/70" : "border-line")}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} className="bg-surface text-ink">
            {o.label}
          </option>
        ))}
      </select>
      {error ? <p className="text-xs text-critical-text">{error}</p> : hint ? <p className="text-xs text-ink-3">{hint}</p> : null}
    </div>
  );
}

export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  format = (v) => String(v),
  className,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
  className?: string;
}) {
  const id = useId();
  const pctPos = ((value - min) / (max - min)) * 100;
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex items-center justify-between gap-3 text-[13px]">
        <label htmlFor={id} className="font-medium text-ink-2">
          {label}
        </label>
        <output htmlFor={id} className="font-mono text-xs text-ink tabular">
          {format(value)}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-violet [&::-webkit-slider-thumb]:size-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-[0_0_0_4px_rgb(139_92_246/0.35)]"
        style={{ background: `linear-gradient(90deg, #8b5cf6 0%, #3b82f6 ${pctPos}%, rgb(255 255 255 / 0.1) ${pctPos}%)` }}
      />
    </div>
  );
}

export function Toggle({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  const id = useId();
  return (
    <div className="flex items-start gap-3">
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn("relative mt-0.5 h-5 w-9 shrink-0 rounded-full border transition-colors duration-300", checked ? "border-violet bg-violet/70" : "border-line-strong bg-white/[0.06]")}
      >
        <span className={cn("absolute top-0.5 size-3.5 rounded-full bg-white transition-transform duration-300", checked ? "translate-x-[18px]" : "translate-x-0.5")} />
      </button>
      <label htmlFor={id} className="cursor-pointer text-sm">
        <span className="font-medium text-ink">{label}</span>
        {hint && <span className="block text-xs text-ink-3">{hint}</span>}
      </label>
    </div>
  );
}
