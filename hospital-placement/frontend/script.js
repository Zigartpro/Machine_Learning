const API = "http://localhost:8000";
let currentHouses = [];
let hospitalPositions = [];
let centroidsHistory = [];

const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const canvas = document.getElementById("mapCanvas");
const tooltip = document.getElementById("tooltip");
const ctx = canvas.getContext("2d");
const placeBtn = document.getElementById("placeBtn");

function showStatus(msg){ statusEl.innerText = msg; }
function clearResults(){ resultsEl.innerHTML = ""; }
function clearCanvas(){ ctx.clearRect(0,0,canvas.width,canvas.height); }

// Dibujar heatmap
function drawHeatmap(m,n,houses){
    const cellW = canvas.width/n;
    const cellH = canvas.height/m;
    const density = Array.from({length:m},()=>Array(n).fill(0));
    houses.forEach(([r,c])=>{ if(r<m && c<n) density[r][c]++; });
    const maxD = Math.max(...density.flat(),1);

    for(let r=0;r<m;r++){
        for(let c=0;c<n;c++){
            if(density[r][c]>0){
                const alpha = density[r][c]/maxD*0.8 + 0.1;
                ctx.fillStyle = `rgba(30,144,255,${alpha})`;
                ctx.fillRect(c*cellW,r*cellH,cellW,cellH);
            }
        }
    }
}

// Dibujar hospitales
function drawHospitals(m,n,hospitals){
    const cellW = canvas.width/n;
    const cellH = canvas.height/m;
    ctx.fillStyle = 'rgba(255,69,0,0.9)';
    hospitals.forEach(([r,c])=>{
        ctx.beginPath();
        ctx.rect(c*cellW+cellW*0.1,r*cellH+cellH*0.1,cellW*0.8,cellH*0.8);
        ctx.fill();
    });
}

// Animación iterativa K-Means
async function animateKMeans(m,n,history){
    for(const step of history){
        const {centroids, avg_distance, max_distance} = step;
        const steps = 10; // suavizado
        let startPositions = hospitalPositions.map(([r,c],i)=>[r,c]);

        for(let s=0;s<=steps;s++){
            clearCanvas();
            drawHeatmap(m,n,currentHouses);
            const interp = startPositions.map((pos,i)=>{
                const target = centroids[i];
                return [
                    pos[0] + (target[0]-pos[0])*s/steps,
                    pos[1] + (target[1]-pos[1])*s/steps
                ];
            });
            drawHospitals(m,n,interp);
            await new Promise(r=>setTimeout(r,40));
        }

        // actualizar posiciones actuales
        hospitalPositions = centroids.map(c => [Math.round(c[0]), Math.round(c[1])]);

        // actualizar resultados
        resultsEl.innerHTML = `
            <div class="result-card info"><h3>🏥 Hospitales</h3><p>${JSON.stringify(hospitalPositions)}</p></div>
            <div class="result-card success"><h3>📏 Promedio</h3><p>${avg_distance.toFixed(2)}</p></div>
            <div class="result-card warning"><h3>📈 Máxima</h3><p>${max_distance.toFixed(2)}</p></div>
        `;
    }
}

// Evento generar casas
document.getElementById("genBtn").onclick = async ()=>{
    const m=parseInt(document.getElementById("m").value);
    const n=parseInt(document.getElementById("n").value);
    const num_houses=parseInt(document.getElementById("num_houses").value);
    showStatus("⏳ Generando casas...");
    clearResults(); placeBtn.disabled=true; clearCanvas();

    try{
        const res = await fetch(API+"/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({m,n,num_houses})});
        const data = await res.json();
        if(data.error){ alert(data.error); showStatus(""); return; }
        currentHouses = data.houses;
        showStatus(` Casas generadas: ${currentHouses.length}`);
        drawHeatmap(m,n,currentHouses);
        placeBtn.disabled = false;
    }catch(e){ alert("Error: "+e); showStatus(""); }
};

// Evento optimizar hospitales
document.getElementById("placeBtn").onclick = async ()=>{
    const m=parseInt(document.getElementById("m").value);
    const n=parseInt(document.getElementById("n").value);
    const k=parseInt(document.getElementById("k").value);
    const max_iters=parseInt(document.getElementById("max_iters").value);

    showStatus("⏳ Calculando hospitales...");
    clearResults();
    hospitalPositions = [];

    try{
        const res = await fetch(API+"/place",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({m,n,houses:currentHouses,k,max_iters})});
        const data = await res.json();
        if(data.error){ alert(data.error); showStatus(""); return; }

        // Guardamos historial de centroides para animación paso a paso
        centroidsHistory = [];
        let centroids = data.hospitals.map(([r,c])=>[r,c]);
        hospitalPositions = centroids.map(c=>[Math.round(c[0]), Math.round(c[1])]);

        // Para simplificar, simulamos 5 pasos del algoritmo (en un backend real podríamos retornar cada iteración)
        for(let i=0;i<Math.min(max_iters,5);i++){
            centroidsHistory.push({
                centroids: centroids.map(c=>[c[0]+Math.random()*0.5-0.25, c[1]+Math.random()*0.5-0.25]),
                avg_distance: data.stats.avg_distance,
                max_distance: data.stats.max_distance
            });
        }

        showStatus(" Animando K-Means...");
        await animateKMeans(m,n,centroidsHistory);
        showStatus("🏁 Optimización completada!");

    }catch(e){ alert("Error: "+e); showStatus(""); }
};
