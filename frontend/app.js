const API = "/api";
let entries = [],
  filtered = [],
  idx = 0,
  dataset = "welllog";
const $ = (id) => document.getElementById(id);

// Init
document.addEventListener("DOMContentLoaded", async () => {
  await loadDatasets();

  // Tabs
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.onclick = () => switchTab(btn.dataset.tab);
  });

  // Dataset change
  $("dataset-select").onchange = (e) => {
    dataset = e.target.value;
    refresh();
  };

  // Actions
  $("btn-download").onclick = () =>
    runAction("/api/download", `Download ${dataset}?`);
  $("btn-split").onclick = () => runAction("/api/split", `Split ${dataset}?`);
  $("btn-ocr").onclick = () => {
    fetch(`${API}/run-ocr`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset,
        model: $("model-select").value,
        mode: $("mode-select").value,
      }),
    })
      .then((r) => r.json())
      .then((d) => {
        alert(d.message);
        refreshLogs();
      });
  };
  $("btn-refresh").onclick = refreshLogs;

  // Viewer
  $("btn-prev").onclick = () => nav(-1);
  $("btn-next").onclick = () => nav(1);
  $("search").oninput = filter;
  $("show-success").onchange = filter;
  $("show-errors").onchange = filter;
  document.onkeydown = (e) => {
    if (e.key === "ArrowLeft") nav(-1);
    if (e.key === "ArrowRight") nav(1);
  };

  refresh();
  refreshLogs();
  setInterval(refresh, 5000);
  setInterval(refreshLogs, 10000);
});

async function runAction(endpoint, msg) {
  if (!confirm(msg)) return;
  const limit = prompt("Limit?", "50");
  if (!limit) return;
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, limit: parseInt(limit) }),
  });
  const data = await res.json();
  alert(data.message);
  refreshLogs();
}

async function loadDatasets() {
  try {
    const data = await (await fetch(`${API}/datasets`)).json();
    $("dataset-select").innerHTML = data.datasets
      .map(
        (d) =>
          `<option value="${d}" ${
            d === data.default ? "selected" : ""
          }>${d}</option>`,
      )
      .join("");
    dataset = data.default;
  } catch (e) {
    console.error(e);
  }
}

function switchTab(tab) {
  document.querySelectorAll(".nav-item").forEach((b) => {
    b.classList.toggle("bg-white/15", b.dataset.tab === tab);
    b.classList.toggle("text-white/70", b.dataset.tab !== tab);
  });
  document.querySelectorAll(".page-section").forEach((s) => {
    s.classList.toggle("hidden", s.id !== tab + "-section");
    s.classList.toggle("flex", s.id === tab + "-section");
  });
}

async function refresh() {
  try {
    const data = await (
      await fetch(`${API}/entries?dataset=${dataset}`)
    ).json();
    entries = Array.isArray(data) ? data : [];
    updateStats();
    filter();

    const st = await (await fetch(`${API}/status`)).json();
    $("status-text").textContent = st.running ? `Running (${st.type})` : "Idle";
    $("status-dot").className = `w-2 h-2 rounded-full ${
      st.running ? "bg-green-500 animate-pulse" : "bg-gray-400"
    }`;
  } catch (e) {
    console.error(e);
  }
}

function updateStats() {
  const total = entries.length;
  const ok = entries.filter((e) => e.status === "success").length;
  $("stat-total").textContent = total;
  $("stat-rate").textContent = total
    ? ((ok / total) * 100).toFixed(0) + "%"
    : "0%";
  $("stat-errors").textContent = total - ok;

  const recent = entries.slice(-10).reverse();
  $("jobs-table").innerHTML =
    recent
      .map(
        (e) => `
        <tr class="border-b">
            <td class="py-2 font-mono">${e.filename}</td>
            <td class="py-2"><span class="px-2 py-0.5 rounded-full text-xs ${
              e.status === "success"
                ? "bg-green-100 text-green-800"
                : "bg-red-100 text-red-800"
            }">${e.status}</span></td>
            <td class="py-2">${e.model || "-"}</td>
            <td class="py-2"><button class="text-blue-600 hover:underline" onclick="view('${
              e.filename
            }')">View</button></td>
        </tr>
    `,
      )
      .join("") ||
    '<tr><td colspan="4" class="py-4 text-center text-gray-400">No data</td></tr>';
}

function filter() {
  const q = $("search").value.toLowerCase();
  const showOk = $("show-success").checked;
  const showErr = $("show-errors").checked;
  filtered = entries.filter((e) => {
    if (q && !e.filename.toLowerCase().includes(q)) return false;
    if (e.status === "success" && !showOk) return false;
    if (e.status !== "success" && !showErr) return false;
    return true;
  });
  $("counter").textContent = `${filtered.length ? idx + 1 : 0}/${
    filtered.length
  }`;
  if (filtered.length) show(idx);
  else clear();
}

function nav(d) {
  if (!filtered.length) return;
  idx = (idx + d + filtered.length) % filtered.length;
  $("counter").textContent = `${idx + 1}/${filtered.length}`;
  show(idx);
}

function show(i) {
  const e = filtered[i];
  if (!e) return;
  $("filename").textContent = e.filename;
  $("image-panel").innerHTML =
    e.status === "success"
      ? `<img src="/images/${e.filename}?dataset=${dataset}" class="max-w-full">`
      : '<div class="text-red-500">Error</div>';
  $("ocr-panel").innerHTML =
    e.status === "success"
      ? `<div class="text-left whitespace-pre-wrap text-sm">${
          e.answer || ""
        }</div>`
      : `<div class="text-red-500">${e.status}</div>`;
}

function clear() {
  $("filename").textContent = "No matches";
  $("image-panel").innerHTML = '<div class="text-gray-400">No file</div>';
  $("ocr-panel").innerHTML = '<div class="text-gray-400">No file</div>';
}

function view(fn) {
  switchTab("viewer");
  const i = filtered.findIndex((e) => e.filename === fn);
  if (i >= 0) {
    idx = i;
    show(i);
  }
}

async function refreshLogs() {
  try {
    const data = await (await fetch(`${API}/logs/execution`)).json();
    $("log-output").textContent = data.logs || "No logs.";
    $("log-output").scrollTop = $("log-output").scrollHeight;
  } catch (e) {}
}

window.view = view;
