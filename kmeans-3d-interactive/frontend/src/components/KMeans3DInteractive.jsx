import React, { useState, useEffect } from "react";
import { Play, RotateCcw, Info } from "lucide-react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const KMeans3DInteractive = () => {
  const [numCasas, setNumCasas] = useState(40);
  const [numHospitales, setNumHospitales] = useState(3);
  const [gridX, setGridX] = useState(100);
  const [gridY, setGridY] = useState(100);
  const [casas, setCasas] = useState([]);
  const [hospitales, setHospitales] = useState([]);
  const [iteracion, setIteracion] = useState(0);
  const [running, setRunning] = useState(false);
  const [asignaciones, setAsignaciones] = useState([]);
  const [logs, setLogs] = useState(["Sistema listo"]);
  const [apiUrl] = useState(
    import.meta.env.VITE_API_URL || "http://localhost:5000"
  );

  // ============================================================
  // FUNCIONES MATEMÁTICAS
  // ============================================================

  const distanciaEuclidiana = (p1, p2) => {
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
  };

  const centroide = (puntos) => {
    if (puntos.length === 0) return null;
    const sumX = puntos.reduce((sum, p) => sum + p.x, 0);
    const sumY = puntos.reduce((sum, p) => sum + p.y, 0);
    return {
      x: sumX / puntos.length,
      y: sumY / puntos.length,
    };
  };

  const addLog = (mensaje) => {
    setLogs((prev) => [...prev.slice(-8), mensaje]);
  };

  // ============================================================
  // LLAMADAS A API
  // ============================================================

  const generarCasas = async (num) => {
    try {
      addLog("📡 Generando casas...");
      const response = await fetch(`${apiUrl}/api/generar-casas`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          num_casas: num,
          grid_x: gridX,
          grid_y: gridY,
        }),
      });

      const data = await response.json();
      if (data.success) {
        setCasas(data.casas);
        setAsignaciones(new Array(num).fill(0));
        addLog(`✓ ${num} casas generadas`);
        return data.casas;
      } else {
        addLog("❌ Error al generar casas");
      }
    } catch (error) {
      addLog(`❌ Error: ${error.message}`);
      console.error(error);
    }
  };

  const inicializarHospitales = async (num, casasData) => {
    try {
      addLog("📡 Inicializando hospitales...");
      const response = await fetch(`${apiUrl}/api/inicializar-hospitales`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          num_hospitales: num,
          casas: casasData,
        }),
      });

      const data = await response.json();
      if (data.success) {
        setHospitales(data.hospitales);
        addLog(`✓ ${num} hospitales inicializados`);
        return data.hospitales;
      } else {
        addLog("❌ Error al inicializar hospitales");
      }
    } catch (error) {
      addLog(`❌ Error: ${error.message}`);
      console.error(error);
    }
  };

  const realizarIteracion = async (casasData, hospitalesData) => {
    try {
      const response = await fetch(`${apiUrl}/api/iteracion-kmeans`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          casas: casasData,
          hospitales: hospitalesData,
        }),
      });

      const data = await response.json();
      if (data.success) {
        setHospitales(data.hospitales);
        setAsignaciones(data.asignaciones);
        setIteracion((prev) => prev + 1);
        addLog(
          `Iteración ${
            iteracion + 1
          }: Movimiento = ${data.movimiento_total.toFixed(3)}`
        );

        if (data.convergencia) {
          addLog("✓ ¡Convergencia alcanzada!");
          setRunning(false);
        }

        return data;
      }
    } catch (error) {
      addLog(`❌ Error: ${error.message}`);
      console.error(error);
    }
  };

  const handleGenerarCasas = async () => {
    const casasData = await generarCasas(numCasas);
    if (casasData) {
      await inicializarHospitales(numHospitales, casasData);
      setIteracion(0);
    }
  };

  const ejecutarAuto = async () => {
    if (casas.length === 0 || hospitales.length === 0) {
      const casasData = await generarCasas(numCasas);
      if (casasData) {
        await inicializarHospitales(numHospitales, casasData);
      }
      return;
    }

    setRunning(true);
    let hospActuales = hospitales;
    let casasActuales = casas;

    for (let i = 0; i < 20; i++) {
      await new Promise((resolve) => setTimeout(resolve, 300));
      const result = await realizarIteracion(casasActuales, hospActuales);
      if (result) {
        hospActuales = result.hospitales;
      } else {
        break;
      }
    }
    setRunning(false);
  };

  const realizarPaso = async () => {
    if (casas.length === 0 || hospitales.length === 0) {
      addLog("❌ Genera casas y hospitales primero");
      return;
    }
    await realizarIteracion(casas, hospitales);
  };

  const reiniciar = () => {
    setIteracion(0);
    setRunning(false);
    setLogs(["Sistema listo"]);
    setAsignaciones([]);
    setCasas([]);
    setHospitales([]);
  };

  // Preparar datos para visualización
  const datosVisualizacion = [
    ...casas.map((casa, idx) => ({
      x: casa.x,
      y: casa.y,
      tipo: "Casa",
      grupo: asignaciones[idx] || 0,
    })),
    ...hospitales.map((h, idx) => ({
      x: h.x,
      y: h.y,
      tipo: `Hospital ${idx + 1}`,
      grupo: "Hospital",
    })),
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-slate-900 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-5xl font-bold text-white mb-2">
            🏥 K-Means 3D Interactivo
          </h1>
          <p className="text-blue-200 text-lg">
            Optimización de ubicación de hospitales
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Panel de Control */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800 rounded-xl p-6 shadow-2xl border border-slate-700">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                <Info size={24} /> Control
              </h2>

              {/* Input Casas */}
              <div className="mb-4">
                <label className="block text-blue-200 font-semibold mb-2">
                  Número de Casas:{" "}
                  <span className="text-yellow-300">{numCasas}</span>
                </label>
                <input
                  type="range"
                  min="10"
                  max="200"
                  value={numCasas}
                  onChange={(e) => setNumCasas(parseInt(e.target.value))}
                  disabled={running}
                  className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <div className="text-xs text-slate-400 mt-1">10 - 200</div>
              </div>

              {/* Input Hospitales */}
              <div className="mb-6">
                <label className="block text-blue-200 font-semibold mb-2">
                  Número de Hospitales:{" "}
                  <span className="text-yellow-300">{numHospitales}</span>
                </label>
                <input
                  type="range"
                  min="2"
                  max="8"
                  value={numHospitales}
                  onChange={(e) => setNumHospitales(parseInt(e.target.value))}
                  disabled={running}
                  className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <div className="text-xs text-slate-400 mt-1">2 - 8</div>
              </div>

              {/* Información */}
              <div className="bg-slate-700 rounded-lg p-4 mb-6">
                <div className="text-sm text-slate-300">
                  <p>
                    <span className="text-green-300 font-semibold">Casas:</span>{" "}
                    {casas.length}
                  </p>
                  <p>
                    <span className="text-red-300 font-semibold">
                      Hospitales:
                    </span>{" "}
                    {hospitales.length}
                  </p>
                  <p>
                    <span className="text-yellow-300 font-semibold">
                      Iteración:
                    </span>{" "}
                    {iteracion}
                  </p>
                </div>
              </div>

              {/* Botones */}
              <button
                onClick={handleGenerarCasas}
                disabled={running}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white font-bold py-3 px-4 rounded-lg mb-3 transition-colors flex items-center justify-center gap-2"
              >
                <RotateCcw size={20} /> Generar
              </button>

              <button
                onClick={ejecutarAuto}
                disabled={running || casas.length === 0}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-slate-600 text-white font-bold py-3 px-4 rounded-lg mb-3 transition-colors flex items-center justify-center gap-2"
              >
                <Play size={20} /> {running ? "Ejecutando..." : "Auto"}
              </button>

              <button
                onClick={realizarPaso}
                disabled={running || casas.length === 0}
                className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-slate-600 text-white font-bold py-3 px-4 rounded-lg mb-3 transition-colors"
              >
                Paso a Paso
              </button>

              <button
                onClick={reiniciar}
                disabled={running}
                className="w-full bg-red-600 hover:bg-red-700 disabled:bg-slate-600 text-white font-bold py-3 px-4 rounded-lg transition-colors"
              >
                Reiniciar
              </button>
            </div>

            {/* Logs */}
            <div className="bg-slate-800 rounded-xl p-4 mt-6 border border-slate-700 max-h-48 overflow-y-auto">
              <h3 className="text-white font-bold mb-3">📋 Eventos</h3>
              <div className="space-y-1">
                {logs.map((log, idx) => (
                  <div
                    key={idx}
                    className="text-xs text-slate-300 font-mono p-1 bg-slate-700 rounded"
                  >
                    {log}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Gráfico 2D */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800 rounded-xl p-6 shadow-2xl border border-slate-700 h-full">
              <h2 className="text-white font-bold mb-4 text-lg">
                📊 Visualización 2D
              </h2>

              {casas.length > 0 ? (
                <ResponsiveContainer width="100%" height={500}>
                  <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    data={datosVisualizacion}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis
                      dataKey="x"
                      type="number"
                      domain={[0, gridX]}
                      stroke="#888"
                    />
                    <YAxis
                      dataKey="y"
                      type="number"
                      domain={[0, gridY]}
                      stroke="#888"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1e293b",
                        border: "1px solid #475569",
                        borderRadius: "8px",
                        color: "#fff",
                      }}
                      cursor={{ strokeDasharray: "3 3" }}
                    />
                    <Legend wrapperStyle={{ color: "#fff" }} />

                    {/* Casas */}
                    <Scatter
                      name="Casas"
                      data={datosVisualizacion.filter((d) => d.tipo === "Casa")}
                      fill="#4ade80"
                      shape="circle"
                      size={60}
                    />

                    {/* Hospitales */}
                    <Scatter
                      name="Hospitales"
                      data={datosVisualizacion.filter((d) =>
                        d.tipo.includes("Hospital")
                      )}
                      fill="#ef4444"
                      shape="diamond"
                      size={200}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-96 flex items-center justify-center text-slate-400">
                  <p className="text-lg">
                    Configura parámetros y presiona "Generar"
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Estadísticas */}
        {casas.length > 0 && (
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800 rounded-lg p-4 border border-green-600">
              <p className="text-green-300 text-sm font-semibold">
                Casas Totales
              </p>
              <p className="text-3xl font-bold text-white">{casas.length}</p>
            </div>

            <div className="bg-slate-800 rounded-lg p-4 border border-red-600">
              <p className="text-red-300 text-sm font-semibold">Hospitales</p>
              <p className="text-3xl font-bold text-white">
                {hospitales.length}
              </p>
            </div>

            <div className="bg-slate-800 rounded-lg p-4 border border-yellow-600">
              <p className="text-yellow-300 text-sm font-semibold">Iteración</p>
              <p className="text-3xl font-bold text-white">{iteracion}</p>
            </div>

            <div className="bg-slate-800 rounded-lg p-4 border border-purple-600">
              <p className="text-purple-300 text-sm font-semibold">
                Promedio Casas/Hospital
              </p>
              <p className="text-3xl font-bold text-white">
                {hospitales.length > 0
                  ? (casas.length / hospitales.length).toFixed(1)
                  : 0}
              </p>
            </div>
          </div>
        )}

        {/* Información del Algoritmo */}
        <div className="mt-6 bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h3 className="text-white font-bold mb-3 text-lg">
            🎓 Cómo funciona K-Means
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="bg-slate-700 rounded p-3">
              <p className="text-blue-300 font-semibold mb-1">
                1️⃣ Inicialización
              </p>
              <p className="text-slate-300">
                Se seleccionan posiciones iniciales aleatorias para los
                hospitales
              </p>
            </div>
            <div className="bg-slate-700 rounded p-3">
              <p className="text-green-300 font-semibold mb-1">2️⃣ Asignación</p>
              <p className="text-slate-300">
                Cada casa se asigna al hospital más cercano
              </p>
            </div>
            <div className="bg-slate-700 rounded p-3">
              <p className="text-yellow-300 font-semibold mb-1">3️⃣ Recálculo</p>
              <p className="text-slate-300">
                Los hospitales se mueven al centro de sus casas asignadas
              </p>
            </div>
          </div>
          <p className="text-slate-300 text-xs mt-4">
            El proceso se repite hasta que los hospitales dejan de moverse
            (convergencia)
          </p>
        </div>
      </div>
    </div>
  );
};

export default KMeans3DInteractive;
