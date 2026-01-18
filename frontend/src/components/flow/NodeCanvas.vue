<script setup>
import { ref, computed, onMounted } from 'vue'
import { VueFlow, useVueFlow } from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Monitor } from 'lucide-vue-next'
import { useSimulationStore } from '../../stores/simulation'
import PipelineNode from './PipelineNode.vue'
import NodeInspector from '../modals/NodeInspector.vue'

const store = useSimulationStore()
const { fitView } = useVueFlow()

const selectedNode = ref(null)

// Nodes are purely computed from the store state
const nodes = computed(() => store.graphNodes)
const edges = computed(() => store.graphEdges)

onMounted(async () => {
  await store.fetchTelemetry()
  setTimeout(() => fitView({ padding: 0.1 }), 200)
})

const handleNodeClick = (e) => {
  selectedNode.value = e.node
}
</script>

<template>
  <div class="vue-flow-wrapper">
    <!-- SCADA Overlay Grid -->
    <div class="absolute inset-0 z-0 pointer-events-none grid-overlay"></div>

    <VueFlow 
      :nodes="nodes" 
      :edges="edges"
      :default-viewport="{ x: 0, y: 0, zoom: 0.8 }"
      :min-zoom="0.1" 
      :max-zoom="2"
      :nodes-draggable="false"
      :nodes-connectable="false"
      :elements-selectable="true"
      :pan-on-scroll="false"
      :zoom-on-scroll="false"
      :zoom-on-double-click="false"
      :zoom-on-pinch="false"
      :pan-on-drag="false"
      @node-click="handleNodeClick"
      @pane-click="selectedNode = null"
    >
      <!-- Technical Grid Background -->
      <Background :variant="'lines'" :size="40" :gap="40" pattern-color="#1f2428" class="bg-[#050505]" />
      
      <template #node-pipeline="props">
        <PipelineNode v-bind="props" />
      </template>

    </VueFlow>

    <!-- System Header -->
    <div class="absolute top-0 left-0 right-0 h-16 bg-[#0d1117]/80 backdrop-blur border-b border-[#30363d] flex items-center justify-between px-6 z-20">
      <div class="flex items-center gap-4">
         <div class="p-2 bg-blue-500/10 rounded border border-blue-500/20">
            <Monitor class="w-5 h-5 text-blue-500" />
         </div>
         <div>
            <h1 class="font-bold text-white text-lg tracking-tight">SYSTEM MONITOR <span class="text-xs text-gray-500 font-mono ml-2">US-EAST-1</span></h1>
            <div class="text-[10px] text-gray-500 font-mono uppercase tracking-widest">TensorGuard Federation Protocol v2.4</div>
         </div>
      </div>
      
      <!-- Global Stats -->
      <div class="flex gap-8">
         <div class="text-right">
             <div class="text-[10px] text-gray-500 uppercase font-bold">Latency</div>
             <div class="text-sm font-mono font-bold text-green-400">
               {{ store.lastMetrics.latencyMs !== null ? `${store.lastMetrics.latencyMs.toFixed(1)}ms` : 'N/A' }}
             </div>
         </div>
         <div class="text-right">
             <div class="text-[10px] text-gray-500 uppercase font-bold">Throughput</div>
             <div class="text-sm font-mono font-bold text-blue-400">
               {{ store.lastMetrics.throughput || 0 }}
             </div>
         </div>
         <div class="text-right">
             <div class="text-[10px] text-gray-500 uppercase font-bold">Active Nodes</div>
             <div class="text-sm font-mono font-bold text-white">{{ store.lastMetrics.activeNodes }}</div>
         </div>
      </div>
    </div>

    <!-- Telemetry Footer -->
    <div class="absolute bottom-0 left-0 right-0 h-12 bg-[#0d1117] border-t border-[#30363d] flex items-center px-6 z-20 justify-between">
       <div class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full" :class="store.roundStatus === 'active' ? 'bg-green-500 animate-pulse' : store.roundStatus === 'degraded' ? 'bg-orange-500 animate-pulse' : 'bg-gray-600'"></div>
          <span class="text-xs font-mono text-gray-400 uppercase">
            {{ store.errorMessage ? 'Telemetry Unavailable' : store.roundStatus === 'active' ? 'Telemetry Streaming' : store.roundStatus === 'degraded' ? 'Telemetry Degraded' : 'System Idle' }}
          </span>
       </div>
       <div class="text-xs font-mono text-gray-600">
          Uptime: 14d 02h 12m
       </div>
    </div>

    <!-- Inspector -->
    <NodeInspector v-if="selectedNode" :node="selectedNode" @close="selectedNode = null" />
  </div>
</template>

<style scoped>
.vue-flow-wrapper {
  width: 100%;
  height: 100%;
  position: relative;
  background: #050505;
  overflow: hidden; /* Lock overflow */
}

/* Subtle Grid Overlay Vignette */
.grid-overlay {
  background: radial-gradient(circle at center, transparent 0%, #000000 120%);
}
</style>

<style>
/* Global VueFlow Overrides for SCADA Theme */
.vue-flow__edge-path {
  stroke: #1f2428;
  stroke-width: 2;
  transition: stroke 0.3s;
}

/* PCB Trace Animation */
.vue-flow__edge.animated .vue-flow__edge-path {
  stroke: #ff6d5a;
  stroke-width: 2;
  stroke-dasharray: 20;
  animation: flowAnimation 2s linear infinite;
  filter: drop-shadow(0 0 4px rgba(255, 109, 90, 0.5));
}

@keyframes flowAnimation {
  from { stroke-dashoffset: 40; }
  to { stroke-dashoffset: 0; }
}

/* Hide Default UI */
.vue-flow__controls, .vue-flow__minimap {
  display: none !important;
}
</style>
