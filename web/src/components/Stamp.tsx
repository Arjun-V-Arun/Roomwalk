import { shortDate } from "@/lib/format";

// Shown only when the scan's one-time code checked out.
export function Stamp({ scannedAt }: { scannedAt?: string | null }) {
  const date = shortDate(scannedAt);
  return (
    <span className="stamp" aria-label={`Code checked${date ? `, filmed ${date}` : ""}`}>
      Code checked
      {date && <small>Filmed {date}</small>}
    </span>
  );
}

export function Trust({ verified, synthetic, scannedAt }: { verified: boolean; synthetic?: boolean; scannedAt?: string | null }) {
  if (synthetic) return <span className="tag tag-demo">Demo room</span>;
  if (verified) return <Stamp scannedAt={scannedAt} />;
  return <span className="tag tag-signal">Not verified</span>;
}
