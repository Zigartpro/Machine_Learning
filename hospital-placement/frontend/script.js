const API = "https://hospitales-backend.onrender.com";
let currentHouses = [];
let hospitalPositions = [];

const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");
const placeBtn = document.getElementById("placeBtn");

function showStatus(msg) {
  statusEl.innerText = msg;
}
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawHeatmap(m, n, houses) {
  const cellW = canvas.width / n;
  const cellH = canvas.height / m;
  ctx.fillStyle = "rgba(30, 144, 255, 0.5)";
  houses.forEach(([r, c]) => {
    ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
  });
}

function drawHospitals(m, n, hospitals) {
  const cellW = canvas.width / n;
  const cellH = canvas.height / m;
  ctx.fillStyle = "red";
  hospitals.forEach(([r, c]) => {
    ctx.beginPath();
    ctx.arc(
      c * cellW + cellW / 2,
      r * cellH + cellH / 2,
      cellW,
      0,
      Math.PI * 2,
    );
    ctx.fill();
  });
}

document.getElementById("genBtn").onclick = async () => {
  const m = document.getElementById("m").value;
  const n = document.getElementById("n").value;
  const num_houses = document.getElementById("num_houses").value;

  showStatus("⏳ Generando...");
  try {
    const res = await fetch(`${API}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ m, n, num_houses }),
    });
    const data = await res.json();
    currentHouses = data.houses;
    clearCanvas();
    drawHeatmap(m, n, currentHouses);
    placeBtn.disabled = false;
    showStatus(`Casas: ${currentHouses.length}`);
  } catch (e) {
    alert("Error al conectar con el servidor");
  }
};

document.getElementById("placeBtn").onclick = async () => {
  const m = document.getElementById("m").value;
  const n = document.getElementById("n").value;
  const k = document.getElementById("k").value;

  showStatus("⏳ Optimizando...");
  try {
    const res = await fetch(`${API}/place`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ m, n, k, houses: currentHouses }),
    });
    const data = await res.json();

    hospitalPositions = data.hospitals;
    clearCanvas();
    drawHeatmap(m, n, currentHouses);
    drawHospitals(m, n, hospitalPositions);

    resultsEl.innerHTML = `
            <p>Distancia Promedio: ${data.stats.avg_distance.toFixed(2)}</p>
            <p>Distancia Máxima: ${data.stats.max_distance.toFixed(2)}</p>
        `;
    showStatus("🏁 Completado");
  } catch (e) {
    alert("Error en la optimización");
  }
};
