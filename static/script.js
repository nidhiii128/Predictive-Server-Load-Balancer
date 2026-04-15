// 🔥 Show loader
document.querySelector("form")?.addEventListener("submit", () => {
    document.getElementById("loader").classList.remove("hidden");
});

window.onload = () => {

    const predictionEl = document.querySelector(".prediction");
    if (!predictionEl) return;

    // 🔵 Circle Animation
    const circle = document.querySelector(".circle");
    const progress = document.querySelector(".progress-circle");

    if (circle && progress) {
        const value = circle.getAttribute("data-value");
        const offset = 440 - (440 * value) / 100;
        progress.style.strokeDashoffset = offset;

        const prediction = predictionEl.innerText.trim();

        if (prediction === "LOW") {
            progress.style.stroke = "#22c55e";
        } else if (prediction === "MEDIUM") {
            progress.style.stroke = "#f59e0b";
        } else {
            progress.style.stroke = "#ef4444";
        }
    }

    // 📊 Get Data Safely
    const chartDataEl = document.getElementById("chartData");
    if (!chartDataEl) return;

    const probLow = parseFloat(chartDataEl.dataset.low) || 0;
    const probMed = parseFloat(chartDataEl.dataset.med) || 0;
    const probHigh = parseFloat(chartDataEl.dataset.high) || 0;

    const dataValues = [probLow, probMed, probHigh];

    // 📊 Bar Chart (Prediction Confidence)
    new Chart(document.getElementById("probChart"), {
        type: "bar",
        data: {
            labels: ["LOW", "MEDIUM", "HIGH"],
            datasets: [{
                label: "Confidence %",
                data: dataValues,
                backgroundColor: ["#22c55e", "#f59e0b", "#ef4444"]
            }]
        },
        options: {
            plugins: {
                legend: {
                    labels: {
                        color: "#ffffff"
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: "#ffffff"
                    },
                    grid: {
                        color: "rgba(255,255,255,0.1)"
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: "#ffffff"
                    },
                    grid: {
                        color: "rgba(255,255,255,0.1)"
                    }
                }
            }
        }
    });

    // 📈 CPU Trend (Simulation)
    const cpuData = Array.from({ length: 10 }, () => Math.floor(Math.random() * 100));

    new Chart(document.getElementById("cpuChart"), {
        type: "line",
        data: {
            labels: ["-10s","-9s","-8s","-7s","-6s","-5s","-4s","-3s","-2s","Now"],
            datasets: [{
                label: "CPU %",
                data: cpuData,
                borderColor: "#38bdf8",
                backgroundColor: "rgba(56,189,248,0.2)",
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            plugins: {
                legend: {
                    labels: {
                        color: "#ffffff"
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: "#ffffff"
                    },
                    grid: {
                        color: "rgba(255,255,255,0.1)"
                    }
                },
                y: {
                    ticks: {
                        color: "#ffffff"
                    },
                    grid: {
                        color: "rgba(255,255,255,0.1)"
                    }
                }
            }
        }
    });

    // 🧾 History Table
    const table = document.querySelector("#historyTable tbody");

    const cpuInput = document.querySelector("input[name='cpu']")?.value || "-";
    const loadInput = document.querySelector("input[name='load']")?.value || "-";

    const prediction = predictionEl.innerText.trim();
    const decision = document.querySelector(".decision-badge")?.innerText || "-";

    if (table) {
        const row = `
            <tr>
                <td>${cpuInput}</td>
                <td>${loadInput}</td>
                <td>${prediction}</td>
                <td>${decision}</td>
            </tr>
        `;
        table.innerHTML += row;
    }

    // 🧠 Insight
    const insightBox = document.getElementById("insightBox");

    if (insightBox) {
        if (prediction === "HIGH") {
            insightBox.innerText = "⚠ High load detected → Immediate scaling required!";
        } else if (prediction === "MEDIUM") {
            insightBox.innerText = "⚡ Moderate load → Monitor closely.";
        } else {
            insightBox.innerText = "✅ System stable.";
        }
    }
};