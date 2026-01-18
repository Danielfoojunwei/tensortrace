<script setup>
import { 
  Settings, Activity, X, Globe, HardDrive, 
  Wifi, Sliders, CheckCircle, Plus, TerminalSquare, Cpu, Trash2,
  FileJson, Play, Monitor, Zap
} from 'lucide-vue-next'
import { ref, onMounted, onUnmounted, computed } from 'vue'

const props = defineProps(['node'])
const emit = defineEmits(['close', 'delete'])

const activeTab = ref('config')

// Dynamic config fields based on node data
const configFields = computed(() => {
    const meta = props.node.data.meta || {}
    return Object.keys(meta).map(key => ({
        key,
        value: meta[key],
        type: typeof meta[key] === 'number' ? 'number' : 'text'
    }))
})

// Simulated live logs
const logs = ref([])
let logInterval = null

const generateLog = () => {
    const logTypes = [
        { level: 'INFO', msg: `Processing batch of ${Math.floor(Math.random() * 100)} records` },
        { level: 'DEBUG', msg: `Step execution time: ${Math.floor(Math.random() * 50)}ms` },
        { level: 'INFO', msg: 'Output verified against schema' },
        { level: 'WARN', msg: 'Memory usage elevated (82%)' },
    ]
    const log = logTypes[Math.floor(Math.random() * logTypes.length)]
    const timestamp = new Date().toISOString().substring(11, 23)
    logs.value.unshift({ ...log, time: timestamp, id: Date.now() })
    if (logs.value.length > 50) logs.value.pop()
}

onMounted(() => {
    for (let i = 0; i < 5; i++) generateLog()
    logInterval = setInterval(generateLog, 2500)
})

onUnmounted(() => {
    if (logInterval) clearInterval(logInterval)
})
</script>

<template>
  <div class="fixed right-0 top-0 bottom-0 w-[500px] bg-[#0d1117] shadow-2xl z-40 flex flex-col border-l border-[#30363d]" @click.stop>
    
    <!-- Header -->
    <div class="h-14 border-b border-[#30363d] flex items-center justify-between px-6 bg-[#161b22]">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded bg-blue-500/10 flex items-center justify-center text-blue-500 border border-blue-500/20">
          <Settings class="w-4 h-4" />
        </div>
        <div>
           <div class="font-bold text-gray-200 text-sm">{{ node.label }}</div>
           <div class="text-[10px] text-gray-500 font-mono">{{ node.id }}</div>
        </div>
      </div>
      <button @click="$emit('close')" class="p-2 hover:bg-[#30363d] text-gray-400 rounded transition-colors">
         <X class="w-4 h-4" />
      </button>
    </div>

    <!-- Tabs -->
    <div class="flex border-b border-[#30363d] px-6 gap-6 text-sm bg-[#161b22]">
        <button 
           @click="activeTab = 'config'"
           class="py-3 border-b-2 transition-colors font-medium"
           :class="activeTab === 'config' ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-500 hover:text-gray-300'"
        >
           Parameters
        </button>
        <button 
           @click="activeTab = 'logs'"
           class="py-3 border-b-2 transition-colors font-medium"
           :class="activeTab === 'logs' ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-500 hover:text-gray-300'"
        >
           Execution Logs
        </button>
        <button 
           @click="activeTab = 'metrics'"
           class="py-3 border-b-2 transition-colors font-medium"
           :class="activeTab === 'metrics' ? 'border-blue-500 text-blue-400' : 'border-transparent text-gray-500 hover:text-gray-300'"
        >
           Metrics
        </button>
    </div>

    <!-- Body -->
    <div class="flex-1 overflow-y-auto p-6 bg-[#0d1117]">
        
        <!-- Parameters Tab -->
        <div v-if="activeTab === 'config'" class="space-y-6">
            <!-- Node Info Card -->
            <div class="bg-[#161b22] p-4 rounded-lg border border-[#30363d]">
               <h4 class="text-xs font-bold text-gray-500 uppercase mb-3 flex items-center gap-2">
                  <Activity class="w-3 h-3" /> Step Details
               </h4>
               <div class="grid grid-cols-2 gap-4">
                  <div>
                     <label class="text-[10px] text-gray-600 uppercase font-bold">Type</label>
                     <div class="font-medium text-sm text-gray-300 mt-0.5">{{ node.data.subtitle || 'Action' }}</div>
                  </div>
                  <div>
                     <label class="text-[10px] text-gray-600 uppercase font-bold">Status</label>
                     <div class="flex items-center gap-1.5 text-sm font-medium mt-0.5" 
                          :class="node.data.status === 'success' ? 'text-green-500' : 'text-blue-500'">
                        <div class="w-2 h-2 rounded-full" :class="node.data.status === 'success' ? 'bg-green-500' : 'bg-blue-500 animate-pulse'"></div>
                        {{ node.data.status ? node.data.status.toUpperCase() : 'IDLE' }}
                     </div>
                  </div>
               </div>
            </div>

             <!-- Live Metric Highlight -->
             <div v-if="node.data.details" class="bg-[#161b22] p-4 rounded-lg border border-[#30363d] flex items-center justify-between">
                <div>
                  <div class="text-xs text-gray-500 uppercase font-bold mb-1">{{ node.data.details.metric }}</div>
                  <div class="text-2xl font-mono font-bold text-white">{{ node.data.details.value }} <span class="text-sm text-gray-500">{{ node.data.details.unit }}</span></div>
                </div>
                <div class="p-3 bg-blue-500/10 rounded-lg text-blue-500">
                    <Zap class="w-6 h-6" />
                </div>
             </div>

            <!-- Configuration -->
            <div class="bg-[#161b22] p-4 rounded-lg border border-[#30363d] space-y-4">
                <h4 class="text-xs font-bold text-gray-500 uppercase mb-1 flex items-center gap-2">
                    <Sliders class="w-3 h-3" /> Step Configuration
                </h4>
                <div class="space-y-3">
                   <!-- Mock Config Fields -->
                   <div>
                      <label class="block text-xs font-medium text-gray-400 mb-1.5">Execution Timeout (ms)</label>
                      <input type="number" value="5000" class="w-full px-3 py-2 bg-[#0d1117] border border-[#30363d] rounded text-sm text-gray-300 outline-none focus:border-blue-500 transition-colors">
                   </div>
                   <div>
                      <label class="block text-xs font-medium text-gray-400 mb-1.5">Retry Policy</label>
                      <select class="w-full px-3 py-2 bg-[#0d1117] border border-[#30363d] rounded text-sm text-gray-300 outline-none focus:border-blue-500 transition-colors">
                         <option>Exponential Backoff</option>
                         <option>Linear</option>
                         <option>None</option>
                      </select>
                   </div>
                </div>
            </div>
        </div>

        <!-- Logs Tab -->
        <div v-if="activeTab === 'logs'" class="h-full flex flex-col">
            <div class="bg-[#050505] text-green-500 font-mono text-xs p-4 rounded-lg border border-[#30363d] h-full overflow-y-auto">
                <div v-for="log in logs" :key="log.id" class="mb-1.5 border-b border-white/5 pb-1">
                    <span class="text-gray-600">[{{ log.time }}]</span>
                    <span :class="log.level === 'WARN' ? 'text-yellow-500' : 'text-green-500'"> {{ log.level }}:</span>
                    <span class="text-gray-300 ml-1">{{ log.msg }}</span>
                </div>
            </div>
        </div>
        
        <!-- Metrics Tab -->
        <div v-if="activeTab === 'metrics'" class="space-y-4">
             <div class="bg-[#161b22] p-4 rounded-lg border border-[#30363d]">
                <h4 class="text-xs text-gray-500 uppercase mb-4">Throughput History</h4>
                <div class="flex items-end gap-1 h-32 w-full pt-4 border-b border-[#30363d]">
                   <div v-for="i in 20" :key="i" 
                        class="flex-1 bg-blue-600/50 rounded-t hover:bg-blue-500 transition-colors"
                        :style="{ height: `${Math.random() * 100}%` }"
                   ></div>
                </div>
             </div>
             
             <div class="bg-[#161b22] p-4 rounded-lg border border-[#30363d]">
                <h4 class="text-xs text-gray-500 uppercase mb-4">Memory Usage</h4>
                 <div class="h-2 bg-[#30363d] rounded-full overflow-hidden">
                    <div class="h-full bg-orange-500 w-[65%]"></div>
                 </div>
                 <div class="flex justify-between mt-2 text-xs">
                    <span class="text-gray-400">Used: 65%</span>
                    <span class="text-gray-400">Total: 4GB</span>
                 </div>
             </div>
        </div>

    </div>

    <!-- Footer -->
    <div class="p-4 border-t border-[#30363d] bg-[#161b22] flex justify-end gap-3">
        <button @click="$emit('close')" class="px-4 py-2 text-sm text-gray-400 hover:text-white hover:bg-[#30363d] rounded transition-colors">
            Close
        </button>
    </div>

  </div>
</template>
