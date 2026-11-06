export function rupees(n?: number | null): string | null {
  if (n === null || n === undefined) return null;
  return `₹${n.toLocaleString("en-IN")}`;
}

export function shortDate(iso?: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric", timeZone: "Asia/Kolkata" });
}

export function megabytes(bytes: number): string {
  return `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;
}

export function basePath(): string {
  return process.env.NEXT_PUBLIC_BASE_PATH || "";
}

export function roomUrl(id: string, file = ""): string {
  return `${basePath()}/rooms/${id}/${file}`;
}
