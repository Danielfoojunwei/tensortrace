<script setup>
import { ref, onMounted } from 'vue'

// 7 Stages of the Unified Pipeline
const stages = [
  { id: 'ingest', label: 'Ingest', x: 50, color: '#f97316' },
  { id: 'clean', label: 'Clean', x: 200, color: '#3b82f6' },
  { id: 'embed', label: 'Embed', x: 350, color: '#a855f7' },
  { id: 'index', label: 'Index', x: 500, color: '#10b981' },
  { id: 'retrieve', label: 'Retrieve', x: 650, color: '#ef4444' },
  { id: 'rank', label: 'Rank', x: 800, color: '#eab308' },
  { id: 'generate', label: 'Generate', x: 950, color: '#f97316' }
]

const paths = ref([])
const particles = ref([])

onMounted(() => {
  // Generate distinct 'threads' for the braid
  for (let i = 0; i < 5; i++) {
    const d = generatePath(i * 10)
    paths.value.push({ d, color: stages[i % stages.length].color })
  }
})

function generatePath(offset) {
  let d = `M 50 ${150 + offset}`
  stages.forEach((s, i) => {
    if (i === 0) return
    const prev = stages[i-1]
    const cp1x = prev.x + (s.x - prev.x) / 2
    const cp1y = 150 + offset
    const cp2x = prev.x + (s.x - prev.x) / 2
    const cp2y = 150 + offset + (i % 2 === 0 ? 20 : -20) // Wiggle
    d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${s.x} ${150 + offset}`
  })
  return d
}
</script>

<template>
  <div class="w-full overflow-x-auto bg-[#0d1117] rounded-lg border border-[#30363d] p-6 relative">
    <h3 class="text-sm font-bold text-gray-400 mb-4 uppercase tracking-wider">Unified 7-Stage Pipeline Braid</h3>
    
    <svg width="1050" height="300" class="w-full h-auto">
      <!-- Connecting Lines (The Braid) -->
      <path v-for="(p, i) in paths" :key="i" 
            :d="p.d" 
            fill="none" 
            :stroke="p.color" 
            stroke-width="2" 
            stroke-opacity="0.3"
      />

      <!-- Active Particles (Data Flow) -->
      <circle v-for="(p, i) in paths" :key="`p-${i}`" r="3" fill="white">
        <animateMotion :path="p.d" dur="3s" repeatCount="indefinite" :begin="`${i * 0.5}s`" />
      </circle>

      <!-- Stage Nodes -->
      <g v-for="stage in stages" :key="stage.id" :transform="`translate(${stage.x}, 150)`">
        <circle r="6" :fill="stage.color" class="drop-shadow-lg" />
        <circle r="12" :stroke="stage.color" stroke-opacity="0.2" fill="none" class="animate-pulse" />
        <text y="30" text-anchor="middle" fill="#8b949e" class="text-xs font-mono">{{ stage.label }}</text>
      </g>
    </svg>
  </div>
</template>
