import { isRouteErrorResponse, Link, useRouteError } from "react-router";

/** Last-resort boundary for render errors so a broken view never blanks the whole app. */
export function RouteError() {
  const err = useRouteError();
  const message = isRouteErrorResponse(err) ? `${err.status} ${err.statusText}` : err instanceof Error ? err.message : "Unknown error";
  return (
    <div className="grid min-h-dvh place-items-center bg-bg p-6 text-ink">
      <div className="max-w-md rounded-2xl border border-critical/40 bg-surface p-6">
        <h1 className="text-xl font-semibold">This view failed to render</h1>
        <p className="mt-2 text-sm text-ink-2">{message}</p>
        <Link to="/" className="mt-4 inline-block text-sm text-cyan underline-offset-4 hover:underline">
          Back to home
        </Link>
      </div>
    </div>
  );
}
