<script setup>
import { computed } from 'vue'
import { Handle, Position } from '@vue-flow/core'
import { 
  Play, Scissors, Layers, Lock, Database, 
  Activity, ShieldCheck, Zap, Server, CheckCircle, Loader2, ArrowRight
} from 'lucide-vue-next'

const props = defineProps({
  id: String,
  data: Object,
  selected: Boolean
})

const styles = computed(() => {
  switch (props.data.status) {
    case 'running': return 'border-orange-500 shadow-[0_0_20px_rgba(255,109,90,0.2)] ring-1 ring-orange-500/50 scale-[1.02]'
    case 'success': return 'border-green-600 shadow-[0_0_10px_rgba(74,222,128,0.1)]'
    case 'error': return 'border-red-600'
    default: return 'border-[#30363d] hover:border-[#58a6ff]/50'
  }
})

const iconColor = computed(() => {
   switch (props.data.type) {
    case 'trigger': return 'text-orange-500 bg-orange-500/10'
    case 'action': return 'text-blue-500 bg-blue-500/10'
    case 'security': return 'text-emerald-500 bg-emerald-500/10'
    case 'aggregator': return 'text-purple-500 bg-purple-500/10'
    default: return 'text-gray-400 bg-gray-800'
  }
})

const iconComponent = computed(() => {
  switch(props.data.icon) {
    case 'play': return Play
    case 'scissors': return Scissors
    case 'layers': return Layers
    case 'lock': return Lock
    case 'database': return Database
    case 'activity': return Activity
    case 'shield': return ShieldCheck
    case 'zap': return Zap
    case 'server': return Server
    default: return Activity
  }
})
</script>

<template>
  <div 
    class="w-[260px] bg-[#161b22] rounded-xl border transition-all duration-300 relative overflow-hidden group"
    :class="styles"
  >
    <!-- Header -->
    <div class="flex items-center p-4 gap-3 border-b border-[#30363d] bg-[#0d1117]/50">
      <div 
        class="w-10 h-10 rounded-lg flex items-center justify-center shrink-0 border border-white/5 transition-transform group-hover:scale-110"
        :class="iconColor"
      >
        <component :is="iconComponent" class="w-5 h-5" />
      </div>
      
      <div class="flex-1 min-w-0">
        <div class="font-bold text-gray-100 text-sm truncate">{{ data.label }}</div>
        <div class="text-[10px] text-gray-500 font-mono truncate uppercase flex items-center gap-1">
          {{ data.subtitle || data.type }}
        </div>
      </div>

       <!-- Status Indicators -->
       <div v-if="data.status === 'running'" class="shrink-0 bg-orange-500/10 p-1.5 rounded-full">
         <Loader2 class="w-4 h-4 text-orange-500 animate-spin" />
       </div>
       <div v-if="data.status === 'success'" class="shrink-0 bg-green-500/10 p-1.5 rounded-full">
         <CheckCircle class="w-4 h-4 text-green-500" />
       </div>
    </div>

    <!-- Body / Metrics -->
    <div class="p-4 bg-[#161b22] relative">
       <!-- Live Metric Display -->
       <div v-if="data.details" class="flex items-end justify-between">
          <div>
            <div class="text-[10px] text-gray-500 uppercase font-semibold tracking-wider mb-0.5">{{ data.details.metric }}</div>
            <div class="text-xl font-bold text-white font-mono flex items-baseline gap-1">
               {{ data.details.value }}
               <span class="text-xs text-gray-500 font-normal">{{ data.details.unit }}</span>
            </div>
          </div>
          
          <!-- Tiny Sparkline Placeholer -->
          <div class="h-6 w-16 flex items-end gap-0.5 opacity-50">
             <div class="w-1 bg-[#30363d] rounded-t-sm h-[40%]"></div>
             <div class="w-1 bg-[#30363d] rounded-t-sm h-[60%]"></div>
             <div class="w-1 bg-[#30363d] rounded-t-sm h-[30%]"></div>
             <div class="w-1 bg-[#30363d] rounded-t-sm h-[80%]"></div>
             <div class="w-1 bg-green-500 rounded-t-sm h-[90%]"></div>
          </div>
       </div>

       <!-- Footer Info (e.g. Iteration) -->
       <div class="mt-3 pt-3 border-t border-[#30363d] flex justify-between items-center text-[10px]">
          <span class="text-gray-500">Last Iteration: <span class="text-gray-300 font-mono">{{ data.round || '-' }}</span></span>
          <ArrowRight class="w-3 h-3 text-gray-600" />
       </div>
    </div>

    <!-- Handles - Invisible but larger hit area -->
    <Handle 
      type="target" 
      :position="Position.Left" 
      class="!w-3 !h-8 !rounded-sm !bg-[#30363d] !border-none !-left-1.5 transition-colors hover:!bg-[#58a6ff]" 
    />
    <Handle 
      type="source" 
      :position="Position.Right" 
      class="!w-3 !h-8 !rounded-sm !bg-[#30363d] !border-none !-right-1.5 transition-colors hover:!bg-[#58a6ff]" 
    />
  </div>
</template>

<style scoped>
div {
  user-select: none;
}
</style>
