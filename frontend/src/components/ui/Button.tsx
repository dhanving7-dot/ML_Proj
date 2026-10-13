import { LoaderCircle } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Link } from "react-router";
import { cn } from "../../lib/format";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const base =
  "group relative inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-full font-medium transition-[transform,background,box-shadow,color,border-color] duration-300 ease-[var(--ease-out-expo)] active:scale-[0.97] disabled:pointer-events-none disabled:opacity-50";

const variants: Record<Variant, string> = {
  primary:
    "bg-gradient-to-r from-violet via-blue to-cyan bg-[length:200%_100%] bg-left text-white shadow-[0_0_0_1px_rgb(255_255_255/0.08),0_10px_30px_-10px_rgb(139_92_246/0.7)] hover:bg-right hover:shadow-[0_0_0_1px_rgb(255_255_255/0.16),0_14px_40px_-8px_rgb(59_130_246/0.75)]",
  secondary: "glass border border-line-strong text-ink hover:border-violet/60 hover:bg-white/[0.06]",
  ghost: "text-ink-2 hover:bg-white/[0.06] hover:text-ink",
  danger: "border border-critical/50 bg-critical/10 text-critical-text hover:bg-critical/20",
};

const sizes: Record<Size, string> = {
  sm: "h-8 px-3.5 text-[13px]",
  md: "h-10 px-5 text-sm",
  lg: "h-12 px-7 text-[15px]",
};

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: ReactNode;
  to?: string;
}

export function Button({ variant = "primary", size = "md", loading, icon, to, className, children, disabled, type = "button", ...rest }: Props) {
  const classes = cn(base, variants[variant], sizes[size], className);
  const content = (
    <>
      {loading ? <LoaderCircle className="size-4 animate-spin" aria-hidden="true" /> : icon}
      <span>{children}</span>
    </>
  );
  if (to) {
    return (
      <Link to={to} className={classes}>
        {content}
      </Link>
    );
  }
  return (
    <button type={type} className={classes} disabled={disabled || loading} aria-busy={loading || undefined} {...rest}>
      {content}
    </button>
  );
}
