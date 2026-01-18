<script setup>
import { ref, onMounted, computed } from 'vue'
import { Activity, Lock, AlertTriangle, Cloud, Zap, Shield, Globe, Database } from 'lucide-vue-next'

const loading = ref(true)
const metrics = ref(null)

const fetchData = async () => {
    try {
        const res = await fetch('/api/v1/forensics/metrics/extended')
        metrics.value = await res.json()
    } catch (e) {
        console.error("Failed to load metrics", e)
    } finally {
        loading.value = false
    }
}

onMounted(fetchData)

// Helper to generate SVG Path for Line/Area charts
const generatePath = (data, key, height, width, isArea=false) => {
    if (!data || data.length === 0) return ''
    const maxVal = Math.max(...data.map(d => d[key])) * 1.1
    const stepX = width / (data.length - 1)
    
    let d = `M 0 ${height - (data[0][key] / maxVal * height)}`
    data.forEach((pt, i) => {
        d += ` L ${i * stepX} ${height - (pt[key] / maxVal * height)}`
    })
    
    if (isArea) {
        d += ` L ${width} ${height} L 0 ${height} Z`
    }
    return d
}

// Compute Pies
const pieSlices = computed(() => {
    if (!metrics.value) return []
    let total = 0
    let startAngle = 0
    return metrics.value.privacy_pie.map(slice => {
        const angle = (slice.value / 100) * 360
        const x1 = 50 + 40 * Math.cos(Math.PI * (startAngle - 90) / 180)
        const y1 = 50 + 40 * Math.sin(Math.PI * (startAngle - 90) / 180)
        const x2 = 50 + 40 * Math.cos(Math.PI * (startAngle + angle - 90) / 180)
        const y2 = 50 + 40 * Math.sin(Math.PI * (startAngle + angle - 90) / 180)
        
        const d = `M 50 50 L ${x1} ${y1} A 40 40 0 ${angle > 180 ? 1 : 0} 1 ${x2} ${y2} Z`
        startAngle += angle
        return { ...slice, d }
    })
})
</script>

<template>
  <div class="space-y-6 pb-12">
    <div class="flex items-center justify-between">
       <div>
         <h2 class="text-2xl font-bold flex items-center gap-2">
            <Activity class="w-6 h-6 text-primary" /> Mission Control
         </h2>
         <span class="text-xs text-gray-400">Real-time Telemetry & Security Observability</span>
       </div>
       <button @click="fetchData" class="text-xs bg-[#111] border border-[#333] px-3 py-1 rounded hover:bg-[#222]">
         Refresh Data
       </button>
    </div>

    <div v-if="loading" class="h-64 flex items-center justify-center text-gray-500 animate-pulse">
        Accessing Secure Metrics...
    </div>

    <div v-else class="grid grid-cols-12 gap-6">

        <!-- 1. System Health Score (Gauge) -->
        <div class="col-span-12 md:col-span-4 bg-[#111] border border-[#333] rounded-lg p-6 flex flex-col items-center justify-center relative overflow-hidden">
             <h3 class="absolute top-4 left-4 text-xs font-bold text-gray-400 uppercase">System Health</h3>
             <div class="relative w-40 h-40 flex items-center justify-center">
                 <svg class="w-full h-full transform -rotate-90">
                     <circle cx="80" cy="80" r="70" stroke="#333" stroke-width="12" fill="none" />
                     <circle cx="80" cy="80" r="70" stroke="#ff5722" stroke-width="12" fill="none"
                             stroke-dasharray="440" :stroke-dashoffset="440 - (440 * metrics.health_score / 100)" 
                             class="transition-all duration-1000 ease-out" />
                 </svg>
                 <div class="absolute inset-0 flex flex-col items-center justify-center">
                     <span class="text-4xl font-bold text-white">{{ metrics.health_score }}</span>
                     <span class="text-[10px] text-green-500 font-mono">OPTIMAL</span>
                 </div>
             </div>
        </div>

        <!-- 2. Privacy Budget Distribution (Pie Chart) -->
        <div class="col-span-12 md:col-span-4 bg-[#111] border border-[#333] rounded-lg p-6">
            <h3 class="text-xs font-bold text-gray-400 uppercase mb-4 flex items-center gap-2">
                <Lock class="w-3 h-3" /> Privacy Budget (ε)
            </h3>
            <div class="flex items-center gap-4">
                <svg viewBox="0 0 100 100" class="w-32 h-32">
                    <path v-for="(slice, i) in pieSlices" :key="i" :d="slice.d" :fill="slice.color" />
                </svg>
                <div class="space-y-2 text-xs">
                    <div v-for="(slice, i) in pieSlices" :key="i" class="flex items-center gap-2">
                        <div class="w-2 h-2 rounded-full" :style="{ background: slice.color }"></div>
                        <span class="text-gray-400">{{ slice.name }}</span>
                        <span class="font-mono text-white">{{ slice.value }}%</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- 3. Bandwidth Usage (Bar Chart) -->
        <div class="col-span-12 md:col-span-4 bg-[#111] border border-[#333] rounded-lg p-6">
            <h3 class="text-xs font-bold text-gray-400 uppercase mb-4 flex items-center gap-2">
                <Globe class="w-3 h-3" /> Regional Bandwidth
            </h3>
            <div class="h-32 flex items-end justify-between gap-2">
                <div v-for="(bar, i) in metrics.bandwidth_bar" :key="i" class="flex-1 flex flex-col items-center gap-1 group">
                    <div class="w-full bg-[#222] rounded-t relative group-hover:bg-[#333] transition-all" 
                         :style="{ height: (bar.mb / 600 * 100) + '%' }">
                         <div class="absolute -top-4 w-full text-center text-[9px] opacity-0 group-hover:opacity-100 transition-opacity">
                             {{ bar.mb }}MB
                         </div>
                    </div>
                    <span class="text-[9px] text-gray-500">{{ bar.region }}</span>
                </div>
            </div>
        </div>

        <!-- 4. Latency Trends (Line Chart) -->
        <div class="col-span-12 md:col-span-8 bg-[#111] border border-[#333] rounded-lg p-6">
             <h3 class="text-xs font-bold text-gray-400 uppercase mb-4 flex items-center gap-2">
                <Zap class="w-3 h-3" /> Latency Trends (24h)
            </h3>
            <div class="relative h-48 w-full border-l border-b border-[#333] bg-black/20">
                <!-- Grid -->
                <div class="absolute inset-0 grid grid-cols-6 grid-rows-4">
                    <div v-for="n in 24" :key="n" class="border-[0.5px] border-[#333]/30"></div>
                </div>
                
                <svg class="absolute inset-0 w-full h-full" preserveAspectRatio="none">
                    <path :d="generatePath(metrics.latency_line, 'compute', 192, 1000)" fill="none" stroke="#ef4444" stroke-width="2" vector-effect="non-scaling-stroke" />
                    <path :d="generatePath(metrics.latency_line, 'encryption', 192, 1000)" fill="none" stroke="#8b5cf6" stroke-width="2" vector-effect="non-scaling-stroke" />
                    <path :d="generatePath(metrics.latency_line, 'network', 192, 1000)" fill="none" stroke="#ec4899" stroke-width="2" vector-effect="non-scaling-stroke" />
                </svg>
            </div>
            <div class="flex gap-4 mt-2 justify-center">
                <span class="text-xs flex items-center gap-1 text-gray-400"><div class="w-2 h-2 rounded-full bg-red-500"></div> Compute</span>
                <span class="text-xs flex items-center gap-1 text-gray-400"><div class="w-2 h-2 rounded-full bg-purple-500"></div> Encryption</span>
                <span class="text-xs flex items-center gap-1 text-gray-400"><div class="w-2 h-2 rounded-full bg-pink-500"></div> Network</span>
            </div>
        </div>

        <!-- 5. Throughput (Area Chart) -->
        <div class="col-span-12 md:col-span-4 bg-[#111] border border-[#333] rounded-lg p-6">
             <h3 class="text-xs font-bold text-gray-400 uppercase mb-4 flex items-center gap-2">
                <Database class="w-3 h-3" /> Expert Throughput
            </h3>
            <div class="relative h-48 w-full border-l border-b border-[#333]">
                <svg class="absolute inset-0 w-full h-full" preserveAspectRatio="none">
                    <path :d="generatePath(metrics.throughput_area, 'Visual', 192, 1000, true)" fill="rgba(34, 197, 94, 0.2)" stroke="#22c55e" stroke-width="2" vector-effect="non-scaling-stroke" />
                    <path :d="generatePath(metrics.throughput_area, 'Manip', 192, 1000, true)" fill="rgba(59, 130, 246, 0.2)" stroke="#3b82f6" stroke-width="2" vector-effect="non-scaling-stroke" />
                </svg>
            </div>
            <div class="flex gap-4 mt-2 justify-center">
                <span class="text-xs flex items-center gap-1 text-gray-400"><div class="w-2 h-2 rounded-full bg-green-500"></div> Visual</span>
                <span class="text-xs flex items-center gap-1 text-gray-400"><div class="w-2 h-2 rounded-full bg-blue-500"></div> Manipulation</span>
            </div>
        </div>

        <!-- 6. Optimization Efficiency (New) -->
        <div class="col-span-12 md:col-span-4 bg-[#111] border border-[#333] rounded-lg p-6">
             <h3 class="text-xs font-bold text-gray-400 uppercase mb-4 flex items-center gap-2">
                <Zap class="w-3 h-3 text-yellow-500" /> Optimization Efficiency
            </h3>
            <div class="space-y-4">
                <div class="flex items-center justify-between border-b border-[#222] pb-2">
                    <span class="text-xs text-gray-500">Bandwidth Saved (Rand-K)</span>
                    <span class="font-mono text-green-400 text-sm">{{ metrics.sparsity_metrics.bandwidth_saved }}%</span>
                </div>
                <div class="flex items-center justify-between border-b border-[#222] pb-2">
                    <span class="text-xs text-gray-500">Compute Speedup (2:4)</span>
                    <span class="font-mono text-blue-400 text-sm">{{ metrics.sparsity_metrics.compute_speedup }}x</span>
                </div>
                <div class="flex items-center justify-between pb-2">
                    <span class="text-xs text-gray-500">Model Reduction</span>
                    <span class="font-mono text-purple-400 text-sm">{{ metrics.sparsity_metrics.model_reduction }}%</span>
                </div>
            </div>
            <!-- Visual Bar Representation -->
            <div class="mt-4 h-2 bg-[#222] rounded-full overflow-hidden flex">
                <div class="h-full bg-green-500" style="width: 33%"></div>
                <div class="h-full bg-blue-500" style="width: 33%"></div>
                <div class="h-full bg-purple-500" style="width: 34%"></div>
            </div>
             <div class="mt-2 flex justify-between text-[9px] text-gray-600">
                <span>Comm.</span>
                <span>Compute</span>
                <span>Storage</span>
            </div>
        </div>

    </div>
  </div>
</template>
