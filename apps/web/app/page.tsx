"use client";

import React, { useMemo, useState } from "react";
import { recommendCitations, recommendSupervisors } from "../lib/api";
import type { CitationItem, SupervisorItem } from "../lib/api";
import { CitationCard, SupervisorCard } from "../components/ResultCard";

export default function Page() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  const [citationsRaw, setCitationsRaw] = useState<CitationItem[]>([]);
  const [supervisors, setSupervisors] = useState<SupervisorItem[]>([]);

  // UI controls (simple)
  const [showDosbing, setShowDosbing] = useState(true);
  const [showSitasi, setShowSitasi] = useState(true);
  const [sortBy, setSortBy] = useState<"relevance" | "year_desc">("relevance");

  // No pagination -> "Load more"
  const [visibleCount, setVisibleCount] = useState(10);

  async function onSearch() {
    const q = query.trim();
    if (q.length < 2) return;

    setLoading(true);
    setVisibleCount(10);

    try {
      // Dosbing dulu, sitasi berikutnya
      const [sup, cit] = await Promise.all([
        recommendSupervisors(q, 10),
        recommendCitations(q, 50), // backend sudah auto-cutoff, ini cuma max cap internal
      ]);

      setSupervisors(sup.results || []);
      setCitationsRaw(cit.results || []);
    } finally {
      setLoading(false);
    }
  }

  const citationsSorted = useMemo(() => {
    const xs = [...citationsRaw];

    xs.sort((a, b) => {
      if (sortBy === "year_desc") {
        const ay = a.tanggal ? Number(String(a.tanggal).slice(0, 4)) : 0;
        const by = b.tanggal ? Number(String(b.tanggal).slice(0, 4)) : 0;
        if (by !== ay) return by - ay;
      }
      // relevance fallback
      return (b.score ?? 0) - (a.score ?? 0);
    });

    return xs;
  }, [citationsRaw, sortBy]);

  const citationsVisible = useMemo(() => {
    return citationsSorted.slice(0, visibleCount);
  }, [citationsSorted, visibleCount]);

  return (
    <div style={{ maxWidth: 980, margin: "0 auto", padding: 16 }}>
      <h1 style={{ fontSize: 22, fontWeight: 900, marginBottom: 12 }}>
        DINUS Research Recommendation
      </h1>

      {/* SEARCH BAR (clean) */}
      <div className="searchCard">
        <div className="searchRow">
          <input
            className="searchInput"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Masukkan ide penelitian Anda..."
            onKeyDown={(e) => {
              if (e.key === "Enter") onSearch();
            }}
          />

          <button className="btnPrimary" onClick={onSearch} disabled={loading}>
            {loading ? "Loading..." : "Cari"}
          </button>
        </div>

        <div className="searchRow2">
          <div className="leftControls">
            <label className="checkItem">
              <input
                type="checkbox"
                checked={showDosbing}
                onChange={(e) => setShowDosbing(e.target.checked)}
              />
              <span>Dosbing</span>
            </label>

            <label className="checkItem">
              <input
                type="checkbox"
                checked={showSitasi}
                onChange={(e) => setShowSitasi(e.target.checked)}
              />
              <span>Sitasi</span>
            </label>
          </div>

          <div className="rightControls">
            <span className="muted" style={{ fontSize: 12 }}>
              Urutkan:
            </span>
            <select
              className="select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              disabled={loading}
            >
              <option value="relevance">Relevansi</option>
              <option value="year_desc">Tahun terbaru</option>
            </select>
          </div>
        </div>
      </div>

      {/* DOSBING FIRST */}
      {showDosbing ? (
        <section style={{ marginTop: 18 }}>
          <div className="sectionHeader">
            <h2 className="sectionTitle">Rekomendasi Dosbing</h2>
            <div className="muted" style={{ fontSize: 12 }}>
              {supervisors.length ? `${supervisors.length} kandidat` : ""}
            </div>
          </div>

          {supervisors.length ? (
            supervisors.map((s, idx) => (
              <SupervisorCard key={`${s.dosen}-${idx}`} item={s} />
            ))
          ) : (
            <div className="muted" style={{ marginTop: 8, fontSize: 13 }}>
              Belum ada hasil. Coba cari dulu.
            </div>
          )}
        </section>
      ) : null}

      {/* CITATIONS NEXT */}
      {showSitasi ? (
        <section style={{ marginTop: 22 }}>
          <div className="sectionHeader">
            <h2 className="sectionTitle">Rekomendasi Sitasi</h2>
            <div className="muted" style={{ fontSize: 12 }}>
              {citationsSorted.length ? `${citationsSorted.length} hasil` : ""}
            </div>
          </div>

          {citationsVisible.length ? (
            <>
              {citationsVisible.map((c, idx) => (
                <CitationCard key={`${c.doc_id}-${idx}`} item={c} />
              ))}

              {citationsVisible.length < citationsSorted.length ? (
                <div style={{ marginTop: 12 }}>
                  <button
                    className="btnPrimary"
                    onClick={() => setVisibleCount((v) => v + 10)}
                  >
                    Load more (+10)
                  </button>
                </div>
              ) : null}
            </>
          ) : (
            <div className="muted" style={{ marginTop: 8, fontSize: 13 }}>
              Belum ada hasil. Coba cari dulu.
            </div>
          )}
        </section>
      ) : null}
    </div>
  );
}
