import React, { useMemo, useState } from "react";
import type { CitationItem, SupervisorItem } from "../lib/api";
import {
  buildWhatsAppShareText,
  copyToClipboard,
  formatCitationAPA,
  formatCitationIEEE,
} from "../lib/citation";

export function CitationCard({ item }: { item: CitationItem }) {
  const absHtml = item.explain?.abstract_html || "";
  const matched = item.explain?.matched_terms || [];

  const [copied, setCopied] = useState<string | null>(null);

  const waLink = useMemo(() => {
    const text = buildWhatsAppShareText(item);
    return `https://wa.me/?text=${encodeURIComponent(text)}`;
  }, [item]);

  async function onCopy(style: "apa" | "ieee") {
    const txt =
      style === "apa" ? formatCitationAPA(item) : formatCitationIEEE(item);
    await copyToClipboard(txt);
    setCopied(style);
    window.setTimeout(() => setCopied(null), 1200);
  }

  return (
    <div className="card" style={{ marginTop: 12 }}>
      {item.url ? (
        <div style={{ fontSize: 16, fontWeight: 700, lineHeight: 1.25 }}>
          <a href={item.url} target="_blank" rel="noreferrer">
            {item.judul || "(Tanpa judul)"}
          </a>
        </div>
      ) : null}

      <div className="muted" style={{ marginTop: 6, fontSize: 13 }}>
        <span className="kbd">score</span> {item.score.toFixed(4)}
        {typeof item.score2 === "number" && (
          <>
            {" "}
            <span className="kbd">score2</span> {item.score2.toFixed(4)}
          </>
        )}
        {"  "}
        {item.tanggal ? <>• {item.tanggal}</> : null}
        {item.source ? <> • {item.source}</> : null}
      </div>

      {/* Actions */}
      <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
        <a className="btn" href={waLink} target="_blank" rel="noreferrer">
          Share WhatsApp
        </a>

        <button className="btn" onClick={() => onCopy("apa")}>
          Copy sitasi (APA){copied === "apa" ? " ✓" : ""}
        </button>

        <button className="btn" onClick={() => onCopy("ieee")}>
          Copy sitasi (IEEE){copied === "ieee" ? " ✓" : ""}
        </button>
      </div>

      {/* Abstrak evidence */}
      {absHtml ? (
        <>
          <hr />
          <div
            className="abstract-evidence"
            dangerouslySetInnerHTML={{ __html: absHtml }}
          />
        </>
      ) : null}

      {/* Matched tokens chips */}
      {matched.length ? (
        <div style={{ marginTop: 10 }}>
          <div className="muted" style={{ fontSize: 12 }}>
            Cocok dengan:
          </div>
          {matched.slice(0, 10).map((t) => (
            <span key={t} className="pill">
              {t}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function SupervisorCard({ item }: { item: SupervisorItem }) {
  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div
        style={{ display: "flex", justifyContent: "space-between", gap: 12 }}
      >
        <div style={{ fontSize: 16, fontWeight: 800 }}>{item.dosen}</div>
        <div className="muted" style={{ fontSize: 13 }}>
          <span className="kbd">score</span> {item.score.toFixed(4)}
          {typeof item.similarity === "number" && (
            <>
              {" "}
              <span className="kbd">sim</span> {item.similarity.toFixed(4)}
            </>
          )}
        </div>
      </div>

      <div className="muted" style={{ marginTop: 6, fontSize: 13 }}>
        Publikasi: {item.pub_count ?? 0}
      </div>

      {item.matched_terms?.length ? (
        <div style={{ marginTop: 10 }}>
          <div className="muted" style={{ fontSize: 12 }}>
            Cocok dengan:
          </div>
          {item.matched_terms.map((t) => (
            <span key={t} className="pill">
              {t}
            </span>
          ))}
        </div>
      ) : null}

      {item.samples?.length ? (
        <>
          <hr />
          <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
            Contoh publikasi dosen:
          </div>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {item.samples.slice(0, 3).map((s, idx) => (
              <li key={idx} style={{ marginBottom: 6, fontSize: 13 }}>
                {s.url ? (
                  <a href={s.url} target="_blank" rel="noreferrer">
                    {s.judul || s.doc_id || "Publikasi"}
                  </a>
                ) : (
                  <span>{s.judul || s.doc_id || "Publikasi"}</span>
                )}
                {s.tanggal ? (
                  <span className="muted"> • {s.tanggal}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}
