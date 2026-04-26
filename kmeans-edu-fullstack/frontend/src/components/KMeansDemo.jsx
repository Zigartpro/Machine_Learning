import React, { useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:5000";

export default function KMeansDemo() {
  const [preset, setPreset] = useState("demo1");
  const [numCasas, setNumCasas] = useState(40);
  const [numHosp, setNumHosp] = useState(3);
  const [casas, setCasas] = useState([]);
  const [centroides, setCentroides] = useState([]);
  const [historia, setHistoria] = useState([]);

  const presets = {
    demo1: { numCasas: 30, numHosp: 3 },
    demo2: { numCasas: 60, numHosp: 4 },
    demo3: { numCasas: 120, numHosp: 5 },
  };

  const aplicarPreset = (key) => {
    const p = presets[key];
    setNumCasas(p.numCasas);
    setNumHosp(p.numHosp);
    setPreset(key);
  };

  async function generar() {
    const res = await fetch(`${API}/api/generar-casas`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_casas: numCasas, grid_x: 100, grid_y: 100 }),
    });
    const data = await res.json();
    setCasas(data.casas);
    setCentroides([]);
    setHistoria([]);
  }

  async function optimizar() {
    const res = await fetch(`${API}/api/optimizar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ casas, num_hospitales: numHosp, max_iter: 20 }),
    });
    const data = await res.json();
    if (data.success) {
      setHistoria(data.historia);
      setCentroides(data.centroides);
    }
  }

  const visualizarIter = (i) => {
    if (!historia.length) return;
    const frame = historia[i];
    const asign = frame.asignaciones;
    const points = casas.map((c, idx) => ({
      x: c[0],
      y: c[1],
      grupo: asign[idx],
    }));
    const cents = frame.centroides.map((c, j) => ({
      x: c[0],
      y: c[1],
      grupo: "h" + j,
    }));
    return { points, cents };
  };

  const [frameIdx, setFrameIdx] = useState(0);

  const current = historia.length
    ? visualizarIter(frameIdx)
    : {
        points: casas.map((c) => ({ x: c[0], y: c[1] })),
        cents: centroides.map((c, i) => ({ x: c[0], y: c[1] })),
      };

  return (
    <div>
      <div className="mb-4 flex gap-3">
        <select
          value={preset}
          onChange={(e) => aplicarPreset(e.target.value)}
          className="p-2 bg-slate-800 rounded"
        >
          <option value="demo1">Preset 1 — 30 casas / 3 hospitales</option>
          <option value="demo2">Preset 2 — 60 casas / 4 hospitales</option>
          <option value="demo3">Preset 3 — 120 casas / 5 hospitales</option>
        </select>

        <input
          type="number"
          value={numCasas}
          onChange={(e) => setNumCasas(parseInt(e.target.value) || 0)}
          className="p-2 bg-slate-800 rounded"
        />
        <input
          type="number"
          value={numHosp}
          onChange={(e) => setNumHosp(parseInt(e.target.value) || 1)}
          className="p-2 bg-slate-800 rounded"
        />

        <button onClick={generar} className="px-3 py-2 bg-green-600 rounded">
          Generar Casas
        </button>
        <button onClick={optimizar} className="px-3 py-2 bg-blue-600 rounded">
          Optimizar
        </button>
      </div>

      <div
        style={{ width: "100%", height: 500 }}
        className="bg-white rounded p-2"
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid />
            <XAxis type="number" dataKey="x" domain={[0, 100]} />
            <YAxis type="number" dataKey="y" domain={[0, 100]} />
            <Tooltip />

            <Scatter name="Casas" data={current.points} fill="#4ade80" />
            <Scatter
              name="Centroides"
              data={current.cents}
              fill="#ef4444"
              shape="diamond"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {historia.length > 0 && (
        <div className="mt-3 flex items-center gap-3">
          <button
            onClick={() => setFrameIdx(Math.max(0, frameIdx - 1))}
            className="px-3 py-2 bg-slate-700 rounded"
          >
            Prev
          </button>
          <div>
            Iteración {frameIdx + 1} / {historia.length}
          </div>
          <button
            onClick={() =>
              setFrameIdx(Math.min(historia.length - 1, frameIdx + 1))
            }
            className="px-3 py-2 bg-slate-700 rounded"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
