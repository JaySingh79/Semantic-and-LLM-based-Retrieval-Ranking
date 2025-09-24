// static/js/main.js

const queryInput = document.getElementById("query");
const searchBtn = document.getElementById("searchBtn");
const bm25Column = document.getElementById("bm25Column");
const sbertColumn = document.getElementById("sbertColumn");
const topkSelect = document.getElementById("topkSelect");
const statusEl = document.getElementById("status");

let controller = null;

const themeToggle = document.getElementById("themeToggle");

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  themeToggle.textContent = theme === "light" ? "🌙" : "☀️";
}

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  setTheme(current === "light" ? "dark" : "light");
});

// On load
window.addEventListener("load", () => {
  queryInput.focus();
  const savedTheme = localStorage.getItem("theme") || "light";
  setTheme(savedTheme);
});

function highlight(text, query) {
  if (!query) return text;
  // escape regex metacharacters from query tokens
  const tokens = query.split(/\s+/).filter(Boolean).map(t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (tokens.length === 0) return text;
  const re = new RegExp("(" + tokens.join("|") + ")", "ig");
  return text.replace(re, '<mark>$1</mark>');
}

function renderList(container, items, query) {
  if (!items || items.length === 0) {
    container.innerHTML = '<p class="text-muted">No results.</p>';
    return;
  }
  const html = items.map(item => {
    const title = item.title || `Doc ${item.doc_idx}`;
    const scoreLabel = item.score !== undefined ? item.score.toFixed(3) : "-";
    const snippet = highlight(item.snippet || "", query);
    return `
      <div class="result-item mb-3">
        <h5 class="mb-1">${item.rank}. ${title} <small class="text-muted">(${scoreLabel})</small></h5>
        <p class="mb-0 small">${snippet}...</p>
        <div class="meta small text-muted mt-1">id: ${item.doc_idx}</div>
        <hr/>
      </div>
    `;
  }).join("\n");
  container.innerHTML = html;
}

async function doSearch() {
  const query = queryInput.value.trim();
  const topk = topkSelect.value;
  if (!query) {
    bm25Column.innerHTML = '<p class="text-danger">❌ Please enter a query.</p>';
    sbertColumn.innerHTML = '<p class="text-danger">❌ Please enter a query.</p>';
    return;
  }

  // Abort previous request if any
  if (controller) controller.abort();
  controller = new AbortController();
  statusEl.textContent = "Searching…";

  // show spinners
  bm25Column.innerHTML = '<div class="d-flex align-items-center"><div class="spinner-border spinner-border-sm me-2" role="status"></div><small>Retrieving BM25...</small></div>';
  sbertColumn.innerHTML = '<div class="d-flex align-items-center"><div class="spinner-border spinner-border-sm me-2" role="status"></div><small>Re-ranking with SBERT...</small></div>';

  try {
    const res = await fetch("/search", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query, topk}),
      signal: controller.signal
    });
    if (!res.ok) {
      const err = await res.json().catch(()=>({error: "Unknown error"}));
      bm25Column.innerHTML = `<p class="text-danger">Error: ${err.error || res.statusText}</p>`;
      sbertColumn.innerHTML = `<p class="text-danger">Error: ${err.error || res.statusText}</p>`;
      statusEl.textContent = "";
      return;
    }
    const data = await res.json();

    renderList(bm25Column, data.bm25, query);
    if (data.sbert) {
      renderList(sbertColumn, data.sbert, query);
    } else if (data.sbert_error) {
      sbertColumn.innerHTML = `<p class="text-warning">SBERT reranker failed: ${data.sbert_error}</p>`;
    } else {
      sbertColumn.innerHTML = `<p class="text-muted">No SBERT results.</p>`;
    }
    statusEl.textContent = `Done — top ${topk}`;
  } catch (err) {
    if (err.name === "AbortError") {
      statusEl.textContent = "Aborted";
      return;
    }
    bm25Column.innerHTML = `<p class="text-danger">Network/Server error: ${err.message}</p>`;
    sbertColumn.innerHTML = `<p class="text-danger">Network/Server error: ${err.message}</p>`;
    statusEl.textContent = "";
  } finally {
    controller = null;
    setTimeout(()=> statusEl.textContent = "", 2500);
  }
}

// wire events
searchBtn.addEventListener("click", doSearch);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    doSearch();
  }
});

// Optional: focus input on load
window.addEventListener("load", () => {
  queryInput.focus();
});
