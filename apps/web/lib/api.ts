export type CitationItem = {
  doc_id: string;
  doc_idx?: number;
  score: number;
  score2?: number;
  judul?: string;
  keyword?: string;
  tanggal?: string;
  url?: string;
  source?: string;
  abstrak?: string;

  explain?: {
    matched_terms: string[];
    abstract_text?: string;
    abstract_html?: string;
  };
};

export type SupervisorItem = {
  dosen: string;
  score: number;
  similarity?: number;
  matched_terms?: string[];
  pub_count?: number;
  samples?: {
    doc_id?: string;
    judul?: string;
    tanggal?: string;
    url?: string;
  }[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

async function postJSON<T>(path: string, body: any): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export async function recommendCitations(query: string, top_k = 10) {
  return postJSON<{
    results: CitationItem[];
    tokens: string[];
    expanded_tokens?: string[];
    auto_k?: number;
    max_top_k?: number;
  }>("/recommend/citations", { query, top_k });
}

export async function recommendSupervisors(query: string, top_k = 10) {
  return postJSON<{ results: SupervisorItem[]; tokens: string[] }>(
    "/recommend/supervisors",
    { query, top_k },
  );
}
