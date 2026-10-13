import { useEffect, useRef } from "react";
import { usePrefersReducedMotion } from "../../hooks/useReducedMotion";

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  hue: number;
}

/**
 * Lightweight network-style particle field on a 2D canvas (no WebGL).
 * - Node count scales with area and is capped; DPR capped at 1.5.
 * - Pauses when off-screen or when the tab is hidden; draws one static frame with reduced motion.
 * - Nodes drift gently and lean toward the pointer.
 */
export function ParticleField({ className }: { className?: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext("2d");
    if (!el || !ctx) return;
    let w = 0;
    let h = 0;
    let nodes: Node[] = [];
    let raf = 0;
    let visible = true;
    const pointer = { x: -9999, y: -9999 };
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);

    const seed = () => {
      const rect = el.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      el.width = Math.round(w * dpr);
      el.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.min(90, Math.round((w * h) / 16000));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.22,
        vy: (Math.random() - 0.5) * 0.22,
        r: Math.random() * 1.4 + 0.6,
        hue: Math.random() < 0.5 ? 258 : Math.random() < 0.6 ? 190 : 220,
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      const link = 120;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < link * link) {
            const alpha = (1 - Math.sqrt(d2) / link) * 0.22;
            ctx.strokeStyle = `hsla(${(a.hue + b.hue) / 2}, 85%, 70%, ${alpha})`;
            ctx.lineWidth = 0.6;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
      for (const n of nodes) {
        const dx = pointer.x - n.x;
        const dy = pointer.y - n.y;
        const near = dx * dx + dy * dy < 160 * 160;
        ctx.fillStyle = `hsla(${n.hue}, 90%, ${near ? 80 : 70}%, ${near ? 0.95 : 0.7})`;
        ctx.beginPath();
        ctx.arc(n.x, n.y, near ? n.r + 0.8 : n.r, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const step = () => {
      for (const n of nodes) {
        const dx = pointer.x - n.x;
        const dy = pointer.y - n.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 200 * 200 && d2 > 1) {
          n.vx += (dx / Math.sqrt(d2)) * 0.004;
          n.vy += (dy / Math.sqrt(d2)) * 0.004;
        }
        n.vx = Math.max(-0.5, Math.min(0.5, n.vx * 0.995));
        n.vy = Math.max(-0.5, Math.min(0.5, n.vy * 0.995));
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < -10) n.x = w + 10;
        if (n.x > w + 10) n.x = -10;
        if (n.y < -10) n.y = h + 10;
        if (n.y > h + 10) n.y = -10;
      }
      draw();
      raf = requestAnimationFrame(step);
    };

    const start = () => {
      cancelAnimationFrame(raf);
      if (reduced) {
        draw();
        return;
      }
      if (visible && document.visibilityState === "visible") raf = requestAnimationFrame(step);
    };

    seed();
    start();

    const ro = new ResizeObserver(() => {
      seed();
      start();
    });
    ro.observe(el);
    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) start();
      else cancelAnimationFrame(raf);
    });
    io.observe(el);
    const onVis = () => start();
    document.addEventListener("visibilitychange", onVis);
    const onMove = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      pointer.x = e.clientX - r.left;
      pointer.y = e.clientY - r.top;
    };
    const onLeave = () => {
      pointer.x = -9999;
      pointer.y = -9999;
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    document.addEventListener("pointerleave", onLeave);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerleave", onLeave);
    };
  }, [reduced]);

  return <canvas ref={canvas} className={className} aria-hidden="true" />;
}
