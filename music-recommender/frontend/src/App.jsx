import React, { useState, useEffect } from "react";
import { Music, Star, TrendingUp, Search, Sparkles } from "lucide-react";

const API_URL = "https://music-recommender-api-pxhn.onrender.com/api";

export default function App() {
  const [canciones, setCanciones] = useState({});
  const [ratings, setRatings] = useState({});
  const [recomendaciones, setRecomendaciones] = useState(null);
  const [loading, setLoading] = useState(false);
  const [busqueda, setBusqueda] = useState("");
  const [generoFiltro, setGeneroFiltro] = useState("Todos");

  useEffect(() => {
    fetch(`${API_URL}/canciones`)
      .then((res) => res.json())
      .then((data) => setCanciones(data))
      .catch((err) => console.error("Error:", err));
  }, []);

  const handleRating = (cancion, valor) => {
    setRatings((prev) => ({
      ...prev,
      [cancion]: valor,
    }));
  };

  const obtenerRecomendaciones = async () => {
    if (Object.keys(ratings).length === 0) {
      alert("Califica al menos una canción");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/recomendar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ratings, top_n: 10 }),
      });
      const data = await response.json();
      setRecomendaciones(data);
    } catch (err) {
      console.error("Error:", err);
      alert("Error al obtener recomendaciones");
    }
    setLoading(false);
  };

  const cancionesFiltradas = () => {
    let todas = [];
    Object.entries(canciones).forEach(([genero, lista]) => {
      if (generoFiltro === "Todos" || genero === generoFiltro) {
        todas.push(...lista.map((c) => ({ nombre: c, genero })));
      }
    });

    if (busqueda) {
      todas = todas.filter((c) =>
        c.nombre.toLowerCase().includes(busqueda.toLowerCase()),
      );
    }

    return todas;
  };

  const colorGenero = (genero) => {
    const colores = {
      Vallenato: "bg-yellow-500",
      Folclor: "bg-green-500",
      Rock: "bg-red-500",
      Pop: "bg-pink-500",
      Urbano: "bg-purple-500",
      Salsa: "bg-orange-500",
    };
    return colores[genero] || "bg-gray-500";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Header */}
      <div className="bg-black bg-opacity-40 backdrop-blur-md border-b border-white border-opacity-20">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <Music className="w-10 h-10 text-cyan-400" />
            <div>
              <h1 className="text-3xl font-bold text-white">
                MusicIA Colombia
              </h1>
              <p className="text-gray-300">
                Sistema de Recomendación Inteligente
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Panel de Canciones */}
          <div className="lg:col-span-2">
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6 border border-white border-opacity-20">
              <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                <Star className="w-6 h-6 text-yellow-400" />
                Califica tus canciones favoritas
              </h2>

              {/* Filtros */}
              <div className="flex gap-4 mb-6">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Buscar canciones..."
                    value={busqueda}
                    onChange={(e) => setBusqueda(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-white bg-opacity-20 border border-white border-opacity-30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  />
                </div>
                <select
                  value={generoFiltro}
                  onChange={(e) => setGeneroFiltro(e.target.value)}
                  className="px-4 py-2 bg-white bg-opacity-20 border border-white border-opacity-30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 cursor-pointer"
                  style={{
                    backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='white'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`,
                    backgroundRepeat: "no-repeat",
                    backgroundPosition: "right 0.5rem center",
                    backgroundSize: "1.5em 1.5em",
                    paddingRight: "2.5rem",
                  }}
                >
                  <option
                    value="Todos"
                    style={{ backgroundColor: "#1e293b", color: "white" }}
                  >
                    Todos los géneros
                  </option>
                  {Object.keys(canciones).map((genero) => (
                    <option
                      key={genero}
                      value={genero}
                      style={{ backgroundColor: "#1e293b", color: "white" }}
                    >
                      {genero}
                    </option>
                  ))}
                </select>
              </div>

              {/* Lista de canciones */}
              <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                {cancionesFiltradas().map(({ nombre, genero }) => (
                  <div
                    key={nombre}
                    className="bg-white bg-opacity-10 rounded-lg p-4 hover:bg-opacity-20 transition"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <p className="text-white font-medium">{nombre}</p>
                        <span
                          className={`inline-block px-2 py-1 rounded-full text-xs text-white mt-1 ${colorGenero(
                            genero,
                          )}`}
                        >
                          {genero}
                        </span>
                      </div>
                      <div className="flex gap-1">
                        {[1, 2, 3, 4, 5].map((star) => (
                          <button
                            key={star}
                            onClick={() => handleRating(nombre, star)}
                            className={`w-8 h-8 rounded-full transition ${
                              ratings[nombre] >= star
                                ? "bg-yellow-400 text-yellow-900"
                                : "bg-gray-600 text-gray-400 hover:bg-gray-500"
                            }`}
                          >
                            {star}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <button
                onClick={obtenerRecomendaciones}
                disabled={loading || Object.keys(ratings).length === 0}
                className="w-full mt-6 bg-gradient-to-r from-cyan-500 to-blue-500 text-white py-4 rounded-xl font-bold text-lg hover:from-cyan-600 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
              >
                {loading ? (
                  "Generando recomendaciones..."
                ) : (
                  <>
                    <Sparkles className="w-6 h-6" />
                    Obtener Recomendaciones ({Object.keys(ratings).length})
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Panel de Recomendaciones */}
          <div className="lg:col-span-1">
            <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-2xl p-6 border border-white border-opacity-20 sticky top-4">
              <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                <TrendingUp className="w-6 h-6 text-green-400" />
                Recomendaciones
              </h2>

              {!recomendaciones ? (
                <div className="text-center py-8">
                  <Music className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-400">
                    Califica algunas canciones para recibir recomendaciones
                    personalizadas
                  </p>
                </div>
              ) : (
                <div>
                  <div className="bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg p-4 mb-4">
                    <p className="text-white text-sm">Tu perfil musical:</p>
                    <p className="text-white text-2xl font-bold">
                      {recomendaciones.clase_predicha}
                    </p>
                  </div>

                  <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                    {recomendaciones.recomendaciones.map((rec, idx) => (
                      <div
                        key={idx}
                        className="bg-white bg-opacity-10 rounded-lg p-3 hover:bg-opacity-20 transition"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <p className="text-white font-medium text-sm">
                              {rec.cancion}
                            </p>
                            <span
                              className={`inline-block px-2 py-0.5 rounded-full text-xs text-white mt-1 ${colorGenero(
                                rec.genero,
                              )}`}
                            >
                              {rec.genero}
                            </span>
                          </div>
                          <div className="ml-2 bg-cyan-500 text-white px-2 py-1 rounded-lg text-xs font-bold">
                            {rec.puntaje.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
