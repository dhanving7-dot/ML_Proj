import { Button } from "../components/ui/Button";

export default function NotFound() {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-center px-4 py-32 text-center">
      <p className="font-mono text-sm text-ink-3">404</p>
      <h1 className="mt-3 text-4xl font-semibold text-ink">This page does not exist</h1>
      <p className="mt-3 text-ink-2">The link may be outdated. Every page of PlacementPredict is reachable from the navigation.</p>
      <Button to="/" className="mt-8">
        Go home
      </Button>
    </div>
  );
}
