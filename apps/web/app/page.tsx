"use client";

import React, { useMemo, useState } from "react";
import { recommendCitations, recommendSupervisors } from "../lib/api";
import { CitationCard, SupervisorCard } from "../components/ResultCard";

type Tab = "citations" | "supervisors";

export default function Page() {
  const [tab, setTab] = useState<Tab>("citations");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [citations, setCitations] = useState<any>(null);
  const [supervisors, setSupervisors] = useState<any>(null);

  const canRun = useMemo(() => query.trim().length >= 2, [query]);

  async function run() {
    if (!canRun) return;
    setLoading(true);
    setErr(null);

    try {
      const [c, s] = await Promise.all([
        recommendCitations(query.trim()),
        recommendSupervisors(query.trim(), Math.min(20)),
      ]);
      setCitations(c);
      setSupervisors(s);
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <div style={{ fontSize: 22, fontWeight: 900, marginBottom: 20 }}>
        Research Recommendation
      </div>

      <div className="card">
        <div className="row">
          <input
            className="input"
            placeholder="Input topik riset atau ide penelitian Anda..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") run();
            }}
          />
        </div>

        <div
          className="row"
          style={{ marginTop: 10, justifyContent: "space-between" }}
        >
          <div className="row">
            <button
              className={`tab ${tab === "citations" ? "tabActive" : ""}`}
              onClick={() => setTab("citations")}
            >
              Sitasi
            </button>
            <button
              className={`tab ${tab === "supervisors" ? "tabActive" : ""}`}
              onClick={() => setTab("supervisors")}
            >
              Dosen
            </button>
          </div>

          <button className="btn" onClick={run} disabled={!canRun || loading}>
            {loading ? "Running..." : "Run"}
          </button>
        </div>

        {err ? (
          <div style={{ marginTop: 12, color: "#ffb4b4", fontSize: 13 }}>
            Error: {err}
          </div>
        ) : null}
      </div>

      <div style={{ marginTop: 16 }}>
        {tab === "citations" ? (
          <>
            <div className="muted" style={{ marginBottom: 10 }}>
              {citations?.tokens?.length ? (
                <>
                  Tokens:{" "}
                  {citations.tokens.slice(0, 18).map((t: string) => (
                    <span key={t} className="pill">
                      {t}
                    </span>
                  ))}
                </>
              ) : null}
              {citations?.expanded_tokens?.length ? (
                <>
                  <div style={{ marginTop: 8 }}>
                    Expanded:{" "}
                    {citations.expanded_tokens.slice(0, 18).map((t: string) => (
                      <span key={t} className="pill">
                        {t}
                      </span>
                    ))}
                  </div>
                </>
              ) : null}
            </div>

            {citations?.results?.length ? (
              citations.results.map((it: any, idx: number) => (
                <CitationCard key={it.doc_id || idx} item={it} />
              ))
            ) : (
              <div className="muted">Belum ada hasil. Klik Run.</div>
            )}
          </>
        ) : (
          <>
            <div className="muted" style={{ marginBottom: 10 }}>
              {supervisors?.tokens?.length ? (
                <>
                  Tokens:{" "}
                  {supervisors.tokens.slice(0, 18).map((t: string) => (
                    <span key={t} className="pill">
                      {t}
                    </span>
                  ))}
                </>
              ) : null}
            </div>

            {supervisors?.results?.length ? (
              supervisors.results.map((it: any, idx: number) => (
                <SupervisorCard key={it.dosen || idx} item={it} />
              ))
            ) : (
              <div className="muted">Belum ada hasil. Klik Run.</div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
