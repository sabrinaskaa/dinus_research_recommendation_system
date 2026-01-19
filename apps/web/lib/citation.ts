// apps/web/lib/citation.ts
import type { CitationItem } from "./api";

function pickAuthors(item: CitationItem): string {
  // coba beberapa field yang mungkin ada di dataset kamu
  const anyItem = item as any;

  const candidates = [
    anyItem.peneliti,
    anyItem.penulis,
    anyItem.authors,
    anyItem.author,
    anyItem.dosen,
    anyItem.pengarang,
  ];

  for (const c of candidates) {
    if (typeof c === "string" && c.trim()) return c.trim();
  }

  return "Unknown author";
}

export function formatCitationAPA(item: CitationItem): string {
  const authors = pickAuthors(item);
  const year = item.tanggal ? String(item.tanggal).slice(0, 4) : "n.d.";
  const title = item.judul?.trim() || "Untitled";
  const source = item.source?.trim() || "UDINUS";
  const url = item.url?.trim() ? ` ${item.url.trim()}` : "";

  return `${authors} (${year}). ${title}. ${source}.${url}`;
}

export function formatCitationIEEE(item: CitationItem): string {
  const authors = pickAuthors(item);
  const year = item.tanggal ? String(item.tanggal).slice(0, 4) : "n.d.";
  const title = item.judul?.trim() || "Untitled";
  const url = item.url?.trim()
    ? `, [Online]. Available: ${item.url.trim()}`
    : "";
  return `${authors}, "${title}," ${year}${url}.`;
}

export async function copyToClipboard(text: string) {
  await navigator.clipboard.writeText(text);
}

export function buildWhatsAppShareText(item: CitationItem): string {
  const title = item.judul?.trim() || "Untitled";
  const url = item.url?.trim() || "";
  const year = item.tanggal ? String(item.tanggal).slice(0, 4) : "";
  return `${title}${year ? ` (${year})` : ""}${url ? `\n${url}` : ""}`;
}
