"use strict";

const byId = (id) => document.getElementById(id);
const numberFormat = new Intl.NumberFormat("zh-CN");
let refreshMilliseconds = 2000;
let selectedCandidate = "";

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  return numberFormat.format(Number(value || 0));
}

function formatScore(value, digits = 1) {
  return value === null || value === undefined
    ? "—"
    : `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "—";
  }
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m ${total % 60}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function stateLabel(state) {
  return {
    pending: "等待",
    running: "运行中",
    complete: "完成",
    failed: "失败",
  }[state] || state || "未知";
}

function renderOverview(snapshot) {
  const percent = Math.min(100, Math.max(0, Number(snapshot.progress_percent || 0)));
  byId("stage-id").textContent = snapshot.stage_id;
  byId("progress-percent").textContent = `${percent.toFixed(1)}%`;
  byId("progress-count").textContent =
    `${formatNumber(snapshot.games_committed)} / ${formatNumber(snapshot.games_total)}`;
  byId("progress-fill").style.width = `${percent}%`;
  byId("throughput").textContent = Number(
    snapshot.effective_games_per_second ?? snapshot.active_games_per_second ?? 0,
  ).toFixed(2);
  byId("eta").textContent = formatDuration(snapshot.eta_seconds);
  byId("remaining").textContent = `${formatNumber(snapshot.games_remaining)} games remaining`;
  byId("resolved").textContent = formatNumber(snapshot.resolved_games);
  byId("unresolved").textContent = `${formatNumber(snapshot.unresolved_games)} unresolved`;
  byId("task-count").textContent =
    `${formatNumber(snapshot.tasks_complete)} / ${formatNumber(snapshot.tasks_total)}`;
  byId("part-count").textContent = `${formatNumber(snapshot.parts_cached)} immutable parts`;
  byId("plan-fingerprint").textContent = `Plan ${snapshot.plan_fingerprint.slice(0, 16)}`;

  const generated = new Date(snapshot.generated_at);
  byId("updated-at").textContent = generated.toLocaleTimeString("zh-CN", {
    hour12: false,
  });
  const failed = snapshot.state === "failed";
  byId("live-dot").className = `live-dot ${failed ? "failed" : "online"}`;
  byId("live-label").textContent = failed ? "任务失败" : stateLabel(snapshot.state);
}

function renderDevices(snapshot) {
  const gpuByIndex = new Map((snapshot.gpus || []).map((gpu) => [String(gpu.index), gpu]));
  const html = (snapshot.devices || []).map((device) => {
    const gpu = gpuByIndex.get(String(device.device_binding));
    const percent = device.games_total
      ? (100 * device.games_committed) / device.games_total
      : 0;
    const memory = gpu
      ? `${(gpu.memory_used_mib / 1024).toFixed(1)} / ${(gpu.memory_total_mib / 1024).toFixed(0)} GB`
      : "—";
    const utilization = gpu ? `${gpu.utilization_percent}%` : "—";
    const power = gpu ? `${gpu.power_watts.toFixed(0)} W` : "—";
    return `
      <article class="device-card panel">
        <div class="device-head">
          <div class="device-title">
            <strong>H200 · GPU ${escapeHtml(device.device_binding)}</strong>
            <span>${gpu ? escapeHtml(gpu.name) : "telemetry unavailable"}</span>
          </div>
          <span class="state-pill ${device.state === "failed" ? "failed" : ""}">
            ${escapeHtml(stateLabel(device.state))}
          </span>
        </div>
        <div class="device-stats">
          <div class="device-stat"><span>EFFECTIVE G/S</span><strong>${Number(device.effective_games_per_second ?? device.active_games_per_second ?? 0).toFixed(2)}</strong></div>
          <div class="device-stat"><span>GPU UTIL</span><strong>${utilization}</strong></div>
          <div class="device-stat"><span>VRAM</span><strong>${memory}</strong></div>
          <div class="device-stat"><span>POWER</span><strong>${power}</strong></div>
        </div>
        <div class="mini-progress"><span style="width:${Math.min(100, percent).toFixed(2)}%"></span></div>
        <div class="device-foot">
          <span>${formatNumber(device.games_committed)} / ${formatNumber(device.games_total)} games</span>
          <span>${formatNumber(device.tasks_complete)} / ${formatNumber(device.tasks_total)} tasks</span>
        </div>
        ${device.error_message ? `<p class="error-banner">${escapeHtml(device.error_message)}</p>` : ""}
      </article>`;
  }).join("");
  byId("device-grid").innerHTML = html || '<div class="empty-state panel">等待设备状态</div>';
}

function wdlu(row) {
  return `<span class="wdlu numeric">
    <span class="win">${formatNumber(row.wins)}</span> /
    <span class="draw">${formatNumber(row.draws)}</span> /
    <span class="loss">${formatNumber(row.losses)}</span> /
    <span class="unresolved">${formatNumber(row.unresolved)}</span>
  </span>`;
}

function renderStandings(snapshot) {
  const rows = snapshot.standings || [];
  byId("standings-body").innerHTML = rows.map((row) => `
    <tr>
      <td class="rank ${row.rank <= 3 && row.games ? "top" : ""}">${row.rank}</td>
      <td><span class="primary-value">${escapeHtml(row.checkpoint)}</span><span class="secondary-value">v${row.checkpoint_version ?? "?"}</span></td>
      <td><span class="deck-number">${escapeHtml(row.deck_hash)}</span><span class="secondary-value">${escapeHtml(row.deck_label)}</span></td>
      <td><span class="primary-value">${formatScore(row.score, 2)}</span><span class="secondary-value">${formatNumber(row.resolved)} resolved</span></td>
      <td class="numeric">${row.ci95_low === null ? "—" : `${formatScore(row.ci95_low)} – ${formatScore(row.ci95_high)}`}</td>
      <td>${wdlu(row)}</td>
      <td class="numeric">${formatScore(row.seat0_score)} / ${formatScore(row.seat1_score)}</td>
      <td class="numeric">${formatNumber(row.games)}</td>
    </tr>`).join("") || '<tr><td colspan="8" class="empty-row">等待首个结果分片</td></tr>';

  const select = byId("candidate-select");
  const candidateIds = new Set(rows.map((row) => row.bundle_id));
  if (!selectedCandidate || !candidateIds.has(selectedCandidate)) {
    selectedCandidate = rows[0]?.bundle_id || "";
  }
  select.innerHTML = rows.map((row) => `
    <option value="${row.bundle_id}" ${row.bundle_id === selectedCandidate ? "selected" : ""}>
      ${escapeHtml(row.checkpoint)} · ${escapeHtml(row.deck_hash)}
    </option>`).join("");
}

function renderCells(snapshot) {
  const rows = (snapshot.cells || []).filter(
    (row) => row.candidate_bundle_id === selectedCandidate,
  );
  byId("cells-body").innerHTML = rows.map((row) => `
    <tr>
      <td>${escapeHtml(row.opponent_checkpoint)}</td>
      <td class="deck-number">${escapeHtml(row.opponent_deck_hash)}</td>
      <td class="primary-value">${formatScore(row.score, 2)}</td>
      <td>${wdlu(row)}</td>
      <td class="numeric">${formatNumber(row.games)}</td>
    </tr>`).join("") || '<tr><td colspan="5" class="empty-row">此候选尚无已提交 cell</td></tr>';
}

function renderTerminalReasons(snapshot) {
  const entries = Object.entries(snapshot.terminal_reasons || {})
    .sort((left, right) => right[1] - left[1]);
  const total = entries.reduce((sum, entry) => sum + Number(entry[1]), 0);
  byId("terminal-list").innerHTML = entries.map(([reason, count]) => {
    const percent = total ? (100 * count) / total : 0;
    return `
      <div class="terminal-row">
        <span>${escapeHtml(reason)}</span>
        <div class="terminal-bar"><span style="width:${percent.toFixed(2)}%"></span></div>
        <strong>${formatNumber(count)} · ${percent.toFixed(1)}%</strong>
      </div>`;
  }).join("") || '<div class="empty-state">等待终局数据</div>';
}

function renderTasks(snapshot) {
  const rows = snapshot.tasks || [];
  byId("tasks-body").innerHTML = rows.map((row) => `
    <tr>
      <td><span class="primary-value">GPU ${row.device_index}</span><span class="secondary-value">${escapeHtml(row.task_short_id)}</span></td>
      <td>${escapeHtml(row.candidate_checkpoint)}</td>
      <td>${escapeHtml(row.opponent_checkpoint)}</td>
      <td><span class="state-pill ${row.state === "failed" ? "failed" : ""}">${escapeHtml(stateLabel(row.state))}</span></td>
      <td class="numeric"><span class="primary-value">${Number(row.progress_percent).toFixed(1)}%</span><span class="secondary-value">${formatNumber(row.games_committed)} / ${formatNumber(row.games_total)}</span></td>
      <td class="numeric">${Number(row.games_per_second || 0).toFixed(2)} g/s</td>
    </tr>`).join("") || '<tr><td colspan="6" class="empty-row">没有计划任务</td></tr>';
}

function render(snapshot) {
  renderOverview(snapshot);
  renderDevices(snapshot);
  renderStandings(snapshot);
  renderCells(snapshot);
  renderTerminalReasons(snapshot);
  renderTasks(snapshot);
}

function showError(message) {
  const banner = byId("error-banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
  byId("live-dot").className = "live-dot offline";
  byId("live-label").textContent = "读取失败";
}

function clearError() {
  byId("error-banner").classList.add("hidden");
}

async function requestJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${text}`);
  }
  return response.json();
}

async function refresh() {
  try {
    const snapshot = await requestJson("/api/snapshot");
    render(snapshot);
    clearError();
  } catch (error) {
    showError(`无法读取实时结果：${error.message}`);
  } finally {
    window.setTimeout(refresh, refreshMilliseconds);
  }
}

async function bootstrap() {
  try {
    const health = await requestJson("/api/health");
    refreshMilliseconds = Math.max(500, Number(health.refresh_seconds) * 1000);
  } catch (error) {
    showError(`WebUI 初始化失败：${error.message}`);
  }
  await refresh();
}

byId("candidate-select").addEventListener("change", (event) => {
  selectedCandidate = event.target.value;
  requestJson("/api/snapshot").then(renderCells).catch((error) => {
    showError(`无法更新 matchup：${error.message}`);
  });
});

bootstrap();
